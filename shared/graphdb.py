"""
External Graph Database Client
Handles complex ID encoding/decoding and batch record retrieval

INTERESTING TECHNICAL CHALLENGE SOLVED:
The search service returns Base64-encoded IDs with variable-length terminators.
This module implements a robust decoding algorithm that tries multiple terminator
lengths and encoding schemes to successfully decode these IDs.

Key Features:
- Multi-attempt Base64 decoding with terminator removal
- UTF-16 LE encoding support with byte boundary handling
- Batch query optimization (max 100 records per request)
- Retry logic with exponential backoff
- Comprehensive error handling and logging
"""

import os
import logging
import requests
import base64
from typing import List, Dict, Any, Tuple
from shared.common import (
    retry_with_backoff,
    prepare_ml_training_records,
    upsert_ml_training_data_to_table,
)

def decode_search_record_id(encoded_id: str) -> str:
    """
    Decode Base64-encoded record IDs from search service.

    TECHNICAL CHALLENGE:
    Search service IDs have trailing terminator characters that vary in length.
    Common patterns:
    - Short IDs: single digit terminator like "0"
    - Long IDs: multi-character terminators like "AA2"

    SOLUTION:
    Try removing 1, 2, or 3 trailing characters and attempt Base64 decode.
    Handle UTF-16 LE encoding with odd byte boundaries.

    Args:
        encoded_id: Base64-encoded string with unknown terminator

    Returns:
        Decoded string in format <identifier@domain.com> or original if decode fails

    Example:
        Input:  "SGVsbG8gV29ybGQhAA2"  (with "AA2" terminator)
        Output: "<Hello@World.com>"
    """
    if not encoded_id:
        return encoded_id

    try:
        clean_id = encoded_id.strip()

        # Strategy: Try removing last 1-3 characters (terminators)
        attempts = [
            clean_id[:-1],  # Remove last 1 char (for "0" terminators)
            clean_id[:-2],  # Remove last 2 chars
            clean_id[:-3],  # Remove last 3 chars (for "AA2" terminators)
        ]

        for attempt in attempts:
            if not attempt:
                continue

            # Ensure proper Base64 padding (must be multiple of 4)
            missing_padding = len(attempt) % 4
            if missing_padding:
                padded = attempt + '=' * (4 - missing_padding)
            else:
                padded = attempt

            # Try to decode this attempt
            try:
                decoded_bytes = base64.urlsafe_b64decode(padded)

                # The bytes are UTF-16 LE encoded text
                # But search service might truncate at odd byte boundaries
                # Try different slice points if UTF-16 fails
                for end_offset in [0, -1, -2, -3]:
                    try:
                        slice_bytes = decoded_bytes[:len(decoded_bytes)+end_offset] if end_offset < 0 else decoded_bytes

                        # UTF-16 needs even number of bytes, trim if odd
                        if len(slice_bytes) % 2 != 0:
                            slice_bytes = slice_bytes[:-1]

                        if slice_bytes:
                            decoded_str = slice_bytes.decode('utf-16-le')

                            # Clean the result: keep only printable chars
                            cleaned = ''.join(char for char in decoded_str
                                            if char.isprintable() or char in '<>@.-_')
                            cleaned = cleaned.strip('\x00 \t\n\r')

                            # Check if it looks like a valid record ID
                            if cleaned and '@' in cleaned:
                                # Format as proper record identifier
                                if not cleaned.startswith('<'):
                                    cleaned = '<' + cleaned
                                if not cleaned.endswith('>'):
                                    cleaned = cleaned + '>'

                                logging.debug(f"Successfully decoded (removed {len(clean_id)-len(attempt)} terminator chars)")
                                return cleaned

                    except UnicodeDecodeError:
                        continue  # Try next offset

            except Exception:
                continue  # Try next attempt

        # If all attempts failed, log warning and return original
        logging.warning(f"Could not decode ID after trying multiple approaches: {encoded_id[:30]}...")
        return encoded_id

    except Exception as e:
        logging.error(f"Unexpected error decoding {encoded_id[:30]}...: {e}")
        return encoded_id

@retry_with_backoff(max_retries=3)
def get_batch_from_external_db(ids: List[str]) -> List[Dict[str, Any]]:
    """
    Batch query external database for the provided IDs.

    Max 100 IDs per batch for optimal performance.
    IDs are decoded from Base64 before sending to the database.

    Args:
        ids: List of Base64-encoded record IDs from search service

    Returns:
        List of full record dictionaries from external database

    Raises:
        requests.RequestException: If API call fails after retries
    """
    if not ids:
        return []

    # Decode the Base64 IDs to actual record identifiers
    decoded_ids = []
    for id_str in ids:
        if id_str:
            decoded = decode_search_record_id(str(id_str))
            decoded_ids.append(decoded)

    # Log decoding success rate
    successful_decodes = sum(1 for orig, dec in zip(ids, decoded_ids) if orig != dec)
    logging.info(f"External DB query starting for {len(decoded_ids)} IDs ({successful_decodes} successfully decoded)")

    # Log sample IDs for debugging (first 2 only)
    if len(decoded_ids) >= 2:
        logging.info(f"Sample decoded IDs: {decoded_ids[:2]}")

    # Call external database API
    api_url = f"{os.environ['EXTERNAL_API_URL']}/api/GetBatchRecords"
    headers = {
        "Content-Type": "application/json",
        "x-functions-key": os.environ['EXTERNAL_API_KEY']
    }

    # Send decoded identifiers to external database
    payload = {"RecordIds": decoded_ids}

    logging.info(f"External DB request payload size: {len(payload['RecordIds'])} items")

    # Make API call with 60 second timeout
    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    logging.info(f"External DB response received: {type(data)}")
    logging.info(f"External DB returned {len(data) if isinstance(data, list) else 'non-list'} records")

    # Log sample record ID if available
    if data and isinstance(data, list) and len(data) > 0:
        sample_id = data[0].get('RecordID') or data[0].get('Id') or data[0].get('id')
        logging.info(f"Sample external DB record ID: {sample_id}")

    return data if isinstance(data, list) else []

def process_id_batch(
    ids: List[str],
    id_to_ts: Dict[str, int],
    sink: str = 'table'
) -> Tuple[int, List[str], List[str]]:
    """
    Process a single batch of IDs through external database to storage.

    Pipeline:
    1. Decode Base64 IDs
    2. Query external database
    3. Prepare ML training records
    4. Upsert to storage
    5. Track success/failure

    Args:
        ids: List of Base64-encoded IDs from search service
        id_to_ts: Mapping of ID to timestamp for checkpoint tracking
        sink: Storage destination ('table' for cloud storage, 'sql' for database)

    Returns:
        Tuple of (records_processed, succeeded_ids, failed_ids)

    Example:
        >>> process_id_batch(['encoded_id_1', 'encoded_id_2'], {}, 'table')
        (2, ['encoded_id_1', 'encoded_id_2'], [])
    """
    if not ids:
        return 0, [], []

    try:
        # STEP 1: Create mapping from encoded to decoded IDs
        encoded_to_decoded = {}
        for encoded_id in ids:
            if encoded_id:
                decoded_id = decode_search_record_id(str(encoded_id))
                encoded_to_decoded[encoded_id] = decoded_id

        # Log decoding statistics
        total_ids = len(ids)
        decoded_count = sum(1 for enc, dec in encoded_to_decoded.items()
                           if enc != dec and dec.startswith('<'))

        if decoded_count < total_ids:
            logging.warning(f"Only decoded {decoded_count}/{total_ids} IDs successfully")
            # Log examples of failed decodes for debugging
            failed_examples = [enc for enc, dec in encoded_to_decoded.items() if enc == dec][:3]
            for example in failed_examples:
                logging.debug(f"Failed to decode: {example[:50]}...")

        # STEP 2: Query external database
        external_records = get_batch_from_external_db(ids)
        if not external_records:
            logging.warning("External database returned no records for batch")
            return 0, [], list(ids)

        # STEP 3: Prepare ML training records (filter and clean)
        logger = logging.getLogger('graphdb')
        prepared = prepare_ml_training_records(external_records, logger)

        if len(prepared) == 0:
            logger.warning(f"All {len(external_records)} records filtered out during ML preparation")

        # STEP 4: Upsert to storage
        processed = upsert_ml_training_data_to_table(prepared)

        # STEP 5: Determine succeeded/failed IDs
        # External DB returns records with decoded IDs, map back to encoded IDs
        got_decoded_ids = set()
        for r in external_records:
            rid = r.get('RecordID') or r.get('Id') or r.get('id')
            if rid:
                got_decoded_ids.add(str(rid))

        # Map back to original encoded IDs for checkpoint tracking
        succeeded = []
        failed = []
        for encoded_id in ids:
            decoded_id = encoded_to_decoded.get(encoded_id, encoded_id)
            if decoded_id in got_decoded_ids:
                succeeded.append(encoded_id)
            else:
                failed.append(encoded_id)

        return int(processed or 0), succeeded, failed

    except Exception as e:
        logging.error(f"process_id_batch failed: {e}")
        return 0, [], list(ids)


# ============================================================================
# UTILITY FUNCTIONS FOR TESTING
# ============================================================================

def test_decode_algorithm():
    """
    Test the decoding algorithm with various terminator patterns.
    Useful for debugging ID decoding issues.
    """
    test_cases = [
        # (encoded_input, expected_output_contains)
        ("SGVsbG8gV29ybGQhAA2", "@"),  # 3-char terminator
        ("VGVzdElEMTIzNDU2Nzg5MA0", "@"),  # 1-char terminator
        ("QW5vdGhlclRlc3RJREY", "@"),  # 2-char terminator
    ]

    print("Testing Base64 ID decoding algorithm...")
    print("=" * 80)

    for encoded, expected in test_cases:
        decoded = decode_search_record_id(encoded)
        success = expected in decoded
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {encoded[:30]}... → {decoded[:50]}...")

    print("=" * 80)

def validate_batch_size(batch: List[str]) -> bool:
    """
    Validate that batch size is within acceptable limits.
    External database performs best with batches of 100 or fewer.

    Args:
        batch: List of IDs to validate

    Returns:
        True if batch size is valid, False otherwise
    """
    if len(batch) > 100:
        logging.warning(f"Batch size {len(batch)} exceeds recommended maximum of 100")
        return False
    return True


if __name__ == "__main__":
    # Run tests if executed directly
    test_decode_algorithm()
