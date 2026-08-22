import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler


SAFE_MAX = 9007199254740991
TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$")
GEN_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")
URI_RE = re.compile(r"^gs://[^/\s]+/\S+$")
ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def byte_key(value):
    return value.encode("utf-8")


def reason_list(values):
    return sorted(set(values), key=byte_key)


def parse_time(value):
    if not isinstance(value, str):
        return None
    match = TS_RE.fullmatch(value)
    if not match:
        return None
    year, month, day, hour, minute, second = map(int, match.group(1, 2, 3, 4, 5, 6))
    fraction = match.group(7) or ""
    zone = match.group(8)
    if second > 59:
        return None
    if zone == "Z":
        zone_text = "+00:00"
    else:
        off_hour = int(zone[1:3])
        off_minute = int(zone[4:6])
        if off_hour > 14 or off_minute > 59 or (off_hour == 14 and off_minute != 0):
            return None
        zone_text = zone
    micros = (fraction + "000")[:3] + "000"
    try:
        parsed = datetime.fromisoformat(
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{micros}{zone_text}"
        )
    except ValueError:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def utc_text(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def canonicalize(value):
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return " ".join(normalized.split())


def crc32c(data):
    crc = 0xFFFFFFFF
    for octet in data:
        crc ^= octet
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return f"{(crc ^ 0xFFFFFFFF):08x}"


def valid_row(value):
    if not isinstance(value, dict) or set(value.keys()) != ROW_KEYS:
        return False
    if not all(isinstance(value[key], str) for key in ("id", "entity", "eventTime", "text")):
        return False
    revision = value["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0 or revision > SAFE_MAX:
        return False
    return parse_time(value["eventTime"]) is not None


def parse_object(obj):
    supplied_uri = obj.get("uri") if isinstance(obj, dict) else None
    output_uri = supplied_uri if isinstance(supplied_uri, str) else None
    reasons = []
    if not isinstance(obj, dict):
        return output_uri, ["CRC32C_INVALID", "GENERATION_INVALID", "SCHEMA_INVALID", "URI_INVALID"], []

    if not isinstance(supplied_uri, str) or URI_RE.fullmatch(supplied_uri) is None:
        reasons.append("URI_INVALID")
    generation = obj.get("generation")
    fetched = obj.get("fetchedGeneration")
    if not isinstance(generation, str) or GEN_RE.fullmatch(generation) is None:
        reasons.append("GENERATION_INVALID")
    if not isinstance(fetched, str) or GEN_RE.fullmatch(fetched) is None:
        reasons.append("GENERATION_INVALID")
    if generation != fetched:
        reasons.append("GENERATION_MISMATCH")
    supplied_crc = obj.get("crc32c")
    crc_valid = isinstance(supplied_crc, str) and CRC_RE.fullmatch(supplied_crc) is not None
    if not crc_valid:
        reasons.append("CRC32C_INVALID")
    content = obj.get("content")
    if crc_valid and isinstance(content, str) and crc32c(content.encode("utf-8")) != supplied_crc:
        reasons.append("CRC32C_MISMATCH")
    if obj.get("schemaId") != "training-v1" or not isinstance(content, str):
        reasons.append("SCHEMA_INVALID")

    rows = []
    nonblank = 0
    if isinstance(content, str):
        for line in content.splitlines():
            if not line.strip():
                continue
            nonblank += 1
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                reasons.append("JSONL_INVALID")
                continue
            if not valid_row(parsed):
                reasons.append("SCHEMA_INVALID")
            else:
                rows.append(parsed)
        if nonblank == 0:
            reasons.append("SCHEMA_INVALID")
    return output_uri, reason_list(reasons), rows


def policy_values(policy):
    if not isinstance(policy, dict):
        return None
    low = parse_time(policy.get("minTime"))
    high = parse_time(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    threshold = float(threshold)
    if low is None or high is None or low > high or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return None
    return low, high, threshold


def word_set(text):
    words, current = set(), []
    for char in text.lower():
        if unicodedata.category(char)[0] in ("L", "N"):
            current.append(char)
        elif current:
            words.add("".join(current))
            current = []
    if current:
        words.add("".join(current))
    return words


def jaccard(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def row_sort_key(row):
    return byte_key(row["id"]), byte_key(compact(row))


def entry_sort_key(field):
    return lambda item: ((item[field] or "").encode("utf-8"), compact(item).encode("utf-8"))


def build_corpus(payload):
    accepted_rows = []
    rejected_objects = []
    lineage = []
    for obj in payload["objects"]:
        uri, codes, rows = parse_object(obj)
        if codes:
            rejected_objects.append({"uri": uri, "reasonCodes": codes})
            continue
        accepted_rows.extend(rows)
        lineage.append({
            "uri": obj["uri"], "generation": obj["generation"],
            "crc32c": obj["crc32c"], "schemaId": obj["schemaId"]
        })

    canonical_rows = []
    for row in accepted_rows:
        canonical_rows.append({
            "id": row["id"],
            "entity": canonicalize(row["entity"]),
            "eventTime": utc_text(parse_time(row["eventTime"])),
            "revision": row["revision"],
            "text": canonicalize(row["text"]),
        })

    groups = {}
    for row in canonical_rows:
        key = compact([row["entity"], row["eventTime"], row["text"]])
        groups.setdefault(key, []).append(row)
    retained = []
    rejected_rows = []
    for rows in groups.values():
        ranked = sorted(rows, key=lambda row: (-row["revision"], byte_key(row["id"])))
        retained.append(ranked[0])
        rejected_rows.extend({"id": row["id"], "reasonCodes": ["DUPLICATE"]} for row in ranked[1:])

    policy = policy_values(payload["policy"])
    candidates = []
    if policy is None:
        rejected_rows.extend({"id": row["id"], "reasonCodes": ["POLICY_INVALID"]} for row in retained)
    else:
        low, high, threshold = policy
        for row in retained:
            instant = parse_time(row["eventTime"])
            if instant < low or instant > high:
                rejected_rows.append({"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
            else:
                first = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0] % 10
                split = "train" if first <= 5 else ("validation" if first <= 7 else "test")
                candidates.append((row, split))

    splits = {"train": [], "validation": [], "test": []}
    train_rows = [row for row, split in candidates if split == "train"]
    train_words = [word_set(row["text"]) for row in train_rows]
    if policy is not None:
        threshold = policy[2]
        for row, split in candidates:
            if split != "train" and any(jaccard(word_set(row["text"]), words) >= threshold for words in train_words):
                rejected_rows.append({"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                splits[split].append(row)

    digests = {}
    for name in ("train", "validation", "test"):
        splits[name].sort(key=row_sort_key)
        artifact = "".join(compact(row) + "\n" for row in splits[name]).encode("utf-8")
        digests[name] = hashlib.sha256(artifact).hexdigest()

    rejected_objects.sort(key=entry_sort_key("uri"))
    rejected_rows.sort(key=entry_sort_key("id"))
    lineage.sort(key=entry_sort_key("uri"))
    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = compact(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
    route = self.path.split("?", 1)[0].rstrip("/") or "/"

    if route not in ("/build-corpus", "/api/index", "/api/index.py"):
        self.send_json(404, {"error": "NOT_FOUND"})
        return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "INVALID_INPUT"})
            return
        if not isinstance(payload, dict) or "policy" not in payload or not isinstance(payload.get("objects"), list):
            self.send_json(400, {"error": "INVALID_INPUT"})
            return
        self.send_json(200, build_corpus(payload))

    def do_GET(self):
        self.send_json(200, {"status": "ok", "endpoint": "POST /build-corpus"})
