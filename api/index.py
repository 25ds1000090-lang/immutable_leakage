import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response


app = FastAPI()

SAFE_MAX = 9007199254740991

TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

GEN_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")
URI_RE = re.compile(r"^gs://[^/]+/.+$")

ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}


def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def reject_json_constant(value):
    raise ValueError("Invalid JSON constant")


class DuplicateKeyError(ValueError):
    pass


def reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(
                "Duplicate JSON object key"
            )

        result[key] = value

    return result


def strict_json_loads(value):
    return json.loads(
        value,
        parse_constant=reject_json_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def byte_key(value):
    return value.encode("utf-8")


def reason_list(values):
    return sorted(
        set(values),
        key=byte_key,
    )


def parse_time(value):
    if not isinstance(value, str):
        return None

    match = TS_RE.fullmatch(value)

    if not match:
        return None

    year, month, day, hour, minute, second = map(
        int,
        match.group(1, 2, 3, 4, 5, 6),
    )

    fraction = match.group(7) or ""
    zone = match.group(8)

    if second > 59:
        return None

    if zone == "Z":
        zone_text = "+00:00"
    else:
        offset_hour = int(zone[1:3])
        offset_minute = int(zone[4:6])

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        if (
            offset_hour == 14
            and offset_minute != 0
        ):
            return None

        zone_text = zone

    milliseconds = (fraction + "000")[:3]
    microseconds = milliseconds + "000"

    try:
        parsed = datetime.fromisoformat(
            f"{year:04d}-{month:02d}-{day:02d}"
            f"T{hour:02d}:{minute:02d}:{second:02d}"
            f".{microseconds}{zone_text}"
        )
    except ValueError:
        return None

    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def utc_text(value):
    return (
        value.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{value.microsecond // 1000:03d}Z"
    )


def canonicalize(value):
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    normalized = normalized.lower().strip()

    return " ".join(normalized.split())


def crc32c(data):
    crc = 0xFFFFFFFF

    for octet in data:
        crc ^= octet

        for _ in range(8):
            if crc & 1:
                crc = (
                    crc >> 1
                ) ^ 0x82F63B78
            else:
                crc >>= 1

    return f"{(crc ^ 0xFFFFFFFF):08x}"


def valid_row(value):
    if not isinstance(value, dict):
        return False

    if set(value.keys()) != ROW_KEYS:
        return False

    for key in (
        "id",
        "entity",
        "eventTime",
        "text",
    ):
        if not isinstance(value[key], str):
            return False

    revision = value["revision"]

    if isinstance(revision, bool):
        return False

    if not isinstance(revision, int):
        return False

    if revision < 0:
        return False

    if revision > SAFE_MAX:
        return False

    if parse_time(value["eventTime"]) is None:
        return False

    return True


def parse_object(obj):
    supplied_uri = (
        obj.get("uri")
        if isinstance(obj, dict)
        else None
    )

    output_uri = (
        supplied_uri
        if isinstance(supplied_uri, str)
        else None
    )

    reasons = []

    if not isinstance(obj, dict):
        return (
            output_uri,
            [
                "CRC32C_INVALID",
                "GENERATION_INVALID",
                "SCHEMA_INVALID",
                "URI_INVALID",
            ],
            [],
        )

    if (
        not isinstance(supplied_uri, str)
        or URI_RE.fullmatch(supplied_uri) is None
    ):
        reasons.append("URI_INVALID")

    generation = obj.get("generation")
    fetched_generation = obj.get(
        "fetchedGeneration"
    )

    generation_valid = (
        isinstance(generation, str)
        and GEN_RE.fullmatch(generation)
        is not None
    )

    fetched_generation_valid = (
        isinstance(fetched_generation, str)
        and GEN_RE.fullmatch(
            fetched_generation
        )
        is not None
    )

    if (
        not generation_valid
        or not fetched_generation_valid
    ):
        reasons.append(
            "GENERATION_INVALID"
        )

    if generation != fetched_generation:
        reasons.append(
            "GENERATION_MISMATCH"
        )

    supplied_crc = obj.get("crc32c")

    crc_valid = (
        isinstance(supplied_crc, str)
        and CRC_RE.fullmatch(supplied_crc)
        is not None
    )

    if not crc_valid:
        reasons.append(
            "CRC32C_INVALID"
        )

    content = obj.get("content")

    if (
        crc_valid
        and isinstance(content, str)
        and crc32c(
            content.encode("utf-8")
        ) != supplied_crc
    ):
        reasons.append(
            "CRC32C_MISMATCH"
        )

    if (
        obj.get("schemaId") != "training-v1"
        or not isinstance(content, str)
    ):
        reasons.append(
            "SCHEMA_INVALID"
        )

    rows = []
    nonblank_count = 0

    if isinstance(content, str):
        for line in content.split("\n"):
            if not line.strip():
                continue

            nonblank_count += 1

            try:
                parsed = strict_json_loads(line)
            except DuplicateKeyError:
                reasons.append(
                    "SCHEMA_INVALID"
                )
                continue
            except (
                json.JSONDecodeError,
                ValueError,
            ):
                reasons.append(
                    "JSONL_INVALID"
                )
                continue

            if not valid_row(parsed):
                reasons.append(
                    "SCHEMA_INVALID"
                )
            else:
                rows.append(parsed)

        if nonblank_count == 0:
            reasons.append(
                "SCHEMA_INVALID"
            )

    return (
        output_uri,
        reason_list(reasons),
        rows,
    )


def policy_values(policy):
    if not isinstance(policy, dict):
        return None

    minimum = parse_time(
        policy.get("minTime")
    )

    maximum = parse_time(
        policy.get("maxTime")
    )

    threshold = policy.get(
        "contaminationThreshold"
    )

    if isinstance(threshold, bool):
        return None

    if not isinstance(
        threshold,
        (int, float),
    ):
        return None

    threshold = float(threshold)

    if minimum is None:
        return None

    if maximum is None:
        return None

    if minimum > maximum:
        return None

    if not math.isfinite(threshold):
        return None

    if threshold < 0 or threshold > 1:
        return None

    return (
        minimum,
        maximum,
        threshold,
    )


def word_set(text):
    words = set()
    current = []

    for character in text.lower():
        category = unicodedata.category(
            character
        )

        if category[0] in ("L", "N"):
            current.append(character)
        elif current:
            words.add("".join(current))
            current = []

    if current:
        words.add("".join(current))

    return words


def jaccard(left, right):
    if not left and not right:
        return 1.0

    return (
        len(left & right)
        / len(left | right)
    )


def row_sort_key(row):
    return (
        row["id"].encode("utf-8"),
        compact(row).encode("utf-8"),
    )


def rejected_object_sort_key(item):
    uri = item["uri"]

    if uri is None:
        uri_bytes = b""
    else:
        uri_bytes = uri.encode("utf-8")

    return (
        uri_bytes,
        compact(item).encode("utf-8"),
    )


def rejected_row_sort_key(item):
    return (
        item["id"].encode("utf-8"),
        compact(item).encode("utf-8"),
    )


def lineage_sort_key(item):
    return (
        item["uri"].encode("utf-8"),
        compact(item).encode("utf-8"),
    )


def build_corpus(payload):
    accepted_rows = []
    rejected_objects = []
    lineage = []

    for obj in payload["objects"]:
        uri, codes, rows = parse_object(obj)

        if codes:
            rejected_objects.append(
                {
                    "uri": uri,
                    "reasonCodes": codes,
                }
            )
            continue

        accepted_rows.extend(rows)

        lineage.append(
            {
                "uri": obj["uri"],
                "generation": obj["generation"],
                "crc32c": obj["crc32c"],
                "schemaId": obj["schemaId"],
            }
        )

    canonical_rows = []

    for row in accepted_rows:
        canonical_rows.append(
            {
                "id": row["id"],
                "entity": canonicalize(
                    row["entity"]
                ),
                "eventTime": utc_text(
                    parse_time(
                        row["eventTime"]
                    )
                ),
                "revision": row["revision"],
                "text": canonicalize(
                    row["text"]
                ),
            }
        )

    duplicate_groups = {}

    for row in canonical_rows:
        deduplication_key = compact(
            [
                row["entity"],
                row["eventTime"],
                row["text"],
            ]
        )

        duplicate_groups.setdefault(
            deduplication_key,
            [],
        ).append(row)

    retained_rows = []
    rejected_rows = []

    for rows in duplicate_groups.values():
        ranked_rows = sorted(
            rows,
            key=lambda row: (
                -row["revision"],
                row["id"].encode("utf-8"),
            ),
        )

        retained_rows.append(
            ranked_rows[0]
        )

        for losing_row in ranked_rows[1:]:
            rejected_rows.append(
                {
                    "id": losing_row["id"],
                    "reasonCodes": [
                        "DUPLICATE"
                    ],
                }
            )

    policy = policy_values(
        payload["policy"]
    )

    split_candidates = []

    if policy is None:
        for row in retained_rows:
            rejected_rows.append(
                {
                    "id": row["id"],
                    "reasonCodes": [
                        "POLICY_INVALID"
                    ],
                }
            )
    else:
        minimum, maximum, threshold = policy

        for row in retained_rows:
            event_time = parse_time(
                row["eventTime"]
            )

            if (
                event_time < minimum
                or event_time > maximum
            ):
                rejected_rows.append(
                    {
                        "id": row["id"],
                        "reasonCodes": [
                            "OUT_OF_WINDOW"
                        ],
                    }
                )
                continue

            entity_hash = hashlib.sha256(
                row["entity"].encode("utf-8")
            ).digest()

            bucket = entity_hash[0] % 10

            if bucket <= 5:
                split_name = "train"
            elif bucket <= 7:
                split_name = "validation"
            else:
                split_name = "test"

            split_candidates.append(
                (
                    row,
                    split_name,
                )
            )

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    train_rows = [
        row
        for row, split_name
        in split_candidates
        if split_name == "train"
    ]

    train_word_sets = [
        word_set(row["text"])
        for row in train_rows
    ]

    if policy is not None:
        threshold = policy[2]

        for (
            row,
            split_name,
        ) in split_candidates:
            contaminated = False

            if split_name != "train":
                current_words = word_set(
                    row["text"]
                )

                for training_words in train_word_sets:
                    similarity = jaccard(
                        current_words,
                        training_words,
                    )

                    if similarity >= threshold:
                        contaminated = True
                        break

            if contaminated:
                rejected_rows.append(
                    {
                        "id": row["id"],
                        "reasonCodes": [
                            "TRAIN_CONTAMINATION"
                        ],
                    }
                )
            else:
                splits[split_name].append(
                    row
                )

    digests = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        splits[split_name].sort(
            key=row_sort_key
        )

        artifact = "".join(
            compact(row) + "\n"
            for row in splits[split_name]
        ).encode("utf-8")

        digests[split_name] = (
            hashlib.sha256(
                artifact
            ).hexdigest()
        )

    for item in rejected_objects:
        item["reasonCodes"] = reason_list(
            item["reasonCodes"]
        )

    for item in rejected_rows:
        item["reasonCodes"] = reason_list(
            item["reasonCodes"]
        )

    rejected_objects.sort(
        key=rejected_object_sort_key
    )

    rejected_rows.sort(
        key=rejected_row_sort_key
    )

    lineage.sort(
        key=lineage_sort_key
    )

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


def json_response(status, value):
    return Response(
        content=compact(value).encode("utf-8"),
        status_code=status,
        media_type="application/json",
    )


@app.get("/")
@app.get("/api/index")
@app.get("/api/index.py")
async def health():
    return json_response(
        200,
        {
            "status": "ok",
            "endpoint": "POST /build-corpus",
        },
    )


@app.post("/build-corpus")
@app.post("/api/index")
@app.post("/api/index.py")
async def corpus_endpoint(
    request: Request,
):
    try:
        raw_body = await request.body()

        payload = strict_json_loads(
            raw_body.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        DuplicateKeyError,
        json.JSONDecodeError,
        ValueError,
    ):
        return json_response(
            400,
            {"error": "INVALID_INPUT"},
        )

    if (
        not isinstance(payload, dict)
        or "policy" not in payload
        or not isinstance(
            payload.get("objects"),
            list,
        )
    ):
        return json_response(
            400,
            {"error": "INVALID_INPUT"},
        )

    return json_response(
        200,
        build_corpus(payload),
    )
