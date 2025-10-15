"""
Shared utilities for data pipeline functions.
Contains common operations for both incremental and backfill functions.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from functools import wraps
import time

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
                        raise e

                    wait_time = backoff_factor ** attempt
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def now_ms() -> int:
    """Get current time in milliseconds"""
    return int(time.time() * 1000)

def minutes_ago_ms(minutes: int) -> int:
    """Get timestamp from N minutes ago in milliseconds"""
    return int((datetime.utcnow() - timedelta(minutes=minutes)).timestamp() * 1000)

def get_table_client(table_name: str, connection_string: Optional[str] = None):
    """Get cloud storage table client with automatic table creation"""
    from azure.data.tables import TableServiceClient

    conn = connection_string or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not conn:
        raise RuntimeError('AZURE_STORAGE_CONNECTION_STRING is not set')

    svc = TableServiceClient.from_connection_string(conn)

    try:
        table_client = svc.get_table_client(table_name)
        table_client.query_entities(max_results=1)
    except Exception:
        try:
            svc.create_table(table_name)
            logging.info(f"Created table: {table_name}")
        except Exception:
            pass

    return svc.get_table_client(table_name)

def prepare_ml_training_records(batch: List[Dict[str, Any]], logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Prepare ML-specific records with data cleaning applied upfront.
    Only stores fields needed for ML training.
    Filters out non-human-reviewed records and excluded PartitionKeys.
    """
    from datetime import datetime

    records = []
    excluded_partition_keys = {'preprocessing', 'sameduplicate', 'unknownduplicate', 'newduplicate', 'pastdue'}

    for item in batch:
        try:
            record_id = item.get('RecordID') or item.get('Id') or item.get('id')
            if not record_id:
                continue

            partition_key = (item.get('PartitionKey') or
                           item.get('partitionKey') or
                           item.get('table') or
                           item.get('Table'))

            # Filter out excluded PartitionKey values
            if partition_key and partition_key in excluded_partition_keys:
                logger.debug(f"Filtering out record with PartitionKey '{partition_key}'")
                continue

            # Create target variable
            action_taken = 0
            if partition_key:
                partition_key_lower = partition_key.lower().strip()
                if partition_key_lower == 'processed':
                    action_taken = 1
                elif partition_key_lower in ['declined', 'trash']:
                    action_taken = 0

            # Check if seen by human - CRITICAL FILTER
            user_name = item.get('userName') or item.get('UserName') or ''
            seen_by_human = bool(user_name and user_name.strip())
            if not seen_by_human:
                logger.debug(f"Skipping non-human-reviewed record: {record_id}")
                continue

            # Build ML-ready record
            ml_record = {
                'RecordID': str(record_id),
                'PartitionKey': partition_key or 'Unknown',
                'action_taken': action_taken,

                # Core ML features
                'timestamp_days': extract_timestamp_days(item),
                'is_duplicate': 1 if item.get('is_duplicate', False) else 0,
                'project_size': float(item.get('project_size', 0) or 0),
                'is_vip': 1 if (item.get('is_vip', False)) else 0,
                'seen_by_human': 1,

                # Division field
                'division': extract_division(item, logger),

                # VIP features
                **extract_vip_features(item),

                # Segment features
                **extract_segment_features(item),

                # Metadata
                'BatchID': f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                'ProcessedAt': datetime.utcnow().isoformat()
            }

            records.append(ml_record)

        except Exception as e:
            logger.error(f"Error preparing ML record: {e}")
            continue

    logger.info(f"ML data preparation: {len(batch)} input records -> {len(records)} ML-ready records")
    return records

def extract_timestamp_days(item: Dict[str, Any]) -> float:
    """Extract timestamp in days since reference date"""
    from datetime import datetime

    actual_date = item.get('timestamp') or item.get('dateReceived')
    if not actual_date:
        return 0.0

    try:
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

        dt = None
        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(actual_date).strip(), fmt)
                break
            except ValueError:
                continue

        if dt:
            reference = datetime(2020, 1, 1)
            return float((dt - reference).total_seconds() / 86400)
    except Exception:
        pass

    return 0.0

def extract_division(item: Dict[str, Any], logger: logging.Logger) -> float:
    """Extract division field (numeric 0-7)"""
    division_raw = item.get('division') or item.get('Division')
    try:
        division = float(division_raw) if division_raw is not None else 0.0
        if division < 0 or division > 7:
            division = 0.0
    except (ValueError, TypeError):
        division = 0.0

    return division

def extract_vip_features(item: Dict[str, Any]) -> Dict[str, int]:
    """Extract premium/priority features from item"""
    vip_str = item.get('vip_details') or item.get('VipDetails') or ''

    # Map old field names to new generic names
    field_mapping = {
        'Blacklisted': 'Excluded',
        'HotelVip': 'Premium_Tier_1',
        'LargeProjectVip': 'Premium_Tier_2',
        'KeyCustomerVip': 'Premium_Tier_3',
        'HistoricalVip': 'Premium_Tier_4',
        'HistoricalVipRating': 'Premium_Score'
    }

    defaults = {
        'Excluded': 0,
        'Premium_Tier_1': 0,
        'Premium_Tier_2': 0,
        'Premium_Tier_3': 0,
        'Premium_Tier_4': 0,
        'Premium_Score': 0
    }

    if not vip_str:
        return defaults

    try:
        vip_dict = json.loads(vip_str) if isinstance(vip_str, str) else vip_str

        result = {}
        # Parse old field names, output new field names
        for old_key, new_key in field_mapping.items():
            if old_key == 'HistoricalVipRating':
                # Handle rating separately (numeric)
                result[new_key] = int(vip_dict.get(old_key, 0) or 0)
            else:
                # Boolean fields
                val = vip_dict.get(old_key, False)
                result[new_key] = 1 if val else 0

        return result
    except Exception:
        return defaults

def extract_segment_features(item: Dict[str, Any]) -> Dict[str, int]:
    """Extract segment/category features (one-hot encoded)"""
    segment_str = item.get('segment') or item.get('Segment') or ''

    result = {
        'Segment_Type_A': 0,
        'Segment_Type_B': 0,
        'Segment_Type_C': 0,
        'Segment_Unknown': 0,
        'Segment_Type_D': 0,
        'Segment_Type_E': 0
    }

    if not segment_str:
        result['Segment_Unknown'] = 1
        return result

    segment_clean = str(segment_str).strip()

    # Map old segment names to new generic types
    if segment_clean == 'Education':
        result['Segment_Type_A'] = 1
    elif segment_clean in ['Entertainment', 'Transit']:
        result['Segment_Type_B'] = 1
    elif segment_clean == 'Hospitality/Retail/Dialysis':
        result['Segment_Type_C'] = 1
    elif segment_clean == 'Healthcare':
        result['Segment_Type_D'] = 1
    elif segment_clean == 'Spaces':
        result['Segment_Type_E'] = 1
    else:
        result['Segment_Unknown'] = 1

    return result

def upsert_ml_training_data_to_table(records: List[Dict[str, Any]]) -> int:
    """Insert ML records into cloud storage"""
    if not records:
        return 0

    base_url = os.environ.get('DATA_STORAGE_FUNCTION_URL')
    function_key = os.environ.get('DATA_STORAGE_FUNCTION_KEY')
    if not base_url or not function_key:
        raise RuntimeError("Missing DATA_STORAGE_FUNCTION_URL or DATA_STORAGE_FUNCTION_KEY")

    headers = {
        "Content-Type": "application/json",
        "x-functions-key": function_key
    }

    url = f"{base_url.rstrip('/')}/api/UpsertMlTrainingData"

    data = {'Items': records}

    try:
        logging.info(f"Inserting {len(records)} records to ML training storage...")
        resp = requests.post(url, json=data, headers=headers, timeout=60)
        resp.raise_for_status()
        body = resp.json() if resp.content else {}
        processed = body.get('processed', len(records)) if isinstance(body, dict) else len(records)
        logging.info(f"Successfully inserted {processed} records")
        return int(processed)
    except Exception as e:
        logging.error(f"Failed inserting records: {e}")
        raise
