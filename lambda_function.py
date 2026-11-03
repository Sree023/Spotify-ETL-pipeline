
import boto3
import pandas as pd
import io
import os
import urllib.parse
import logging
import posixpath

# =====================================================
# 1. LOGGER CONFIGURATION
# =====================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# =====================================================
# 2. AWS CLIENTS
# =====================================================

s3 = boto3.client("s3")
glue = boto3.client("glue")


# =====================================================
# 3. CONFIGURATION
# =====================================================

BUCKET_NAME = "telco-users-data-s3"

LANDING_PREFIX = "landing/"
PROCESSED_PREFIX = "processed_files/"
REJECTED_PREFIX = "rejected/"

CRAWLER_NAME = os.environ.get(
    "CRAWLER_NAME",
    "glue-cra"
)

REQUIRED_COLUMNS = [
    col.strip()
    for col in os.environ.get(
        "REQUIRED_COLUMNS", ""
    ).split(",")
    if col.strip()
]

NULL_VALUES = ["", "NULL", "NONE", "NAN", "NA", "N/A"]


# =====================================================
# 4. EXTRACT FILE NAME
# =====================================================

def get_filename(source_key):

    filename = posixpath.basename(source_key)

    return posixpath.splitext(filename)[0]


# =====================================================
# 5. READ CSV FROM S3
# =====================================================

def read_csv_from_s3(bucket, key):

    logger.info("Reading source file: s3://%s/%s", bucket, key)

    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    content = response["Body"].read()

    df = pd.read_csv(
        io.BytesIO(content),
        dtype=str,
        keep_default_na=False
    )

    if df.empty:
        raise ValueError("CSV contains no data records")

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if any(not column for column in df.columns):
        raise ValueError("Empty column name found")

    if not df.columns.is_unique:
        raise ValueError("Duplicate column names found")

    logger.info("Total input records: %s", len(df))

    return df


# =====================================================
# 6. DATA QUALITY VALIDATION
# =====================================================

def validate_data(df):

    logger.info("Starting data validation")

    # Check that configured required columns exist in the source file.
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Reject the complete row if ANY column contains a null-like value.
    # This includes empty strings, whitespace-only values, NULL, NONE,
    # NAN, NA and N/A, regardless of whether the column is required.
    null_mask = pd.Series(False, index=df.index)

    for column in df.columns:

        values = (
            df[column]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
        )

        null_mask |= values.isin(NULL_VALUES)

    rejected_df = df[null_mask].copy()
    valid_df = df[~null_mask].copy()

    logger.info("Rows containing nulls (rejected): %s", len(rejected_df))
    logger.info("Rows without nulls (valid): %s", len(valid_df))

    return valid_df, rejected_df


# =====================================================
# 7. REMOVE DUPLICATES
# =====================================================

def remove_duplicates(df):

    before_count = len(df)

    df = df.drop_duplicates(
        keep="first"
    ).copy()

    duplicates_removed = before_count - len(df)

    logger.info(
        "Duplicates removed: %s",
        duplicates_removed
    )

    return df, duplicates_removed


# =====================================================
# 9. SAVE DATA AS PARQUET
# =====================================================

def save_parquet(bucket, output_key, df):

    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        engine="pyarrow",
        index=False
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )

    logger.info(
        "Parquet successfully saved: s3://%s/%s",
        bucket,
        output_key
    )


# =====================================================
# 10. SAVE REJECTED RECORDS
# =====================================================

def save_rejected_records(bucket, source_key, df):

    if df.empty:
        return None

    filename = get_filename(source_key)

    rejected_key = (
        REJECTED_PREFIX
        + filename
        + "_rejected.parquet"
    )

    save_parquet(
        bucket,
        rejected_key,
        df
    )

    return rejected_key


# =====================================================
# 11. SAVE PROCESSED RECORDS
# =====================================================

def save_processed_data(bucket, source_key, df):

    filename = get_filename(source_key)

    output_key = (
        PROCESSED_PREFIX
        + filename
        + ".parquet"
    )

    save_parquet(
        bucket,
        output_key,
        df
    )

    return output_key


# =====================================================
# 12. ARCHIVE FAILED SOURCE FILE
# =====================================================

def save_failed_file(bucket, source_key):

    filename = posixpath.basename(source_key)

    failed_key = (
        REJECTED_PREFIX
        + filename
        + ".failed"
    )

    s3.copy_object(
        Bucket=bucket,
        CopySource={
            "Bucket": bucket,
            "Key": source_key
        },
        Key=failed_key
    )

    logger.info(
        "Failed source archived: s3://%s/%s",
        bucket,
        failed_key
    )


# =====================================================
# 13. START GLUE CRAWLER
# =====================================================

def start_glue_crawler():

    try:

        logger.info(
            "Starting Glue Crawler: %s",
            CRAWLER_NAME
        )

        glue.start_crawler(
            Name=CRAWLER_NAME
        )

        logger.info("Glue Crawler started successfully")

        return "STARTED"

    except glue.exceptions.CrawlerRunningException:

        logger.warning(
            "Glue Crawler is already running"
        )

        return "ALREADY_RUNNING"


# =====================================================
# 14. MAIN ETL PROCESS
# =====================================================

def process_file(bucket, key):

    logger.info("========== ETL STARTED ==========")

    try:

        # Read CSV
        df = read_csv_from_s3(bucket, key)

        total_rows = len(df)

        # Validate data
        valid_df, rejected_df = validate_data(df)

        # Save rejected records
        rejected_key = save_rejected_records(
            bucket,
            key,
            rejected_df
        )

        # Remove duplicates
        valid_df, duplicates_removed = remove_duplicates(
            valid_df
        )

        # Null-containing rows have already been moved to rejected/.
        # Do not replace nulls with 0; only clean records continue.

        # Save processed Parquet
        output_key = None

        if not valid_df.empty:

            output_key = save_processed_data(
                bucket,
                key,
                valid_df
            )

        else:

            logger.warning("No valid records available")

        report = {
            "source_file": key,
            "processed_file": output_key,
            "total_rows": total_rows,
            "processed_rows": len(valid_df),
            "rejected_rows": len(rejected_df),
            "duplicates_removed": duplicates_removed,
            "null_rows_rejected": len(rejected_df),
            "rejected_file": rejected_key
        }

        logger.info("ETL REPORT: %s", report)

        logger.info("========== ETL COMPLETED ==========")

        return report

    except Exception:

        logger.exception(
            "ETL failed for s3://%s/%s",
            bucket,
            key
        )

        try:

            save_failed_file(
                bucket,
                key
            )

        except Exception:

            logger.exception(
                "Failed to archive source file"
            )

        raise


# =====================================================
# 15. LAMBDA HANDLER - S3 OBJECT CREATE EVENTS
# =====================================================

def lambda_handler(event, context):

    logger.info("========== LAMBDA STARTED ==========")

    results = []
    crawler_required = False
    processing_errors = []

    records = event.get("Records", [])

    logger.info(
        "Total event records received: %s",
        len(records)
    )

    for record in records:

        try:

            if record.get("eventSource") != "aws:s3":
                logger.info("Skipping non-S3 event")
                continue

            bucket = record["s3"]["bucket"]["name"]

            key = urllib.parse.unquote_plus(
                record["s3"]["object"]["key"]
            )

            logger.info("Received file: s3://%s/%s", bucket, key)

            # Validate bucket
            if bucket != BUCKET_NAME:
                logger.warning("Unexpected bucket: %s", bucket)
                continue

            # Process only landing folder
            if not key.startswith(LANDING_PREFIX):
                logger.info("Skipping file outside landing/")
                continue

            # Process only CSV files
            if not key.lower().endswith(".csv"):
                logger.info("Skipping non-CSV file")
                continue

            # Execute ETL
            result = process_file(bucket, key)

            results.append({
                "file": key,
                "status": "SUCCESS",
                "details": result
            })

            if result.get("processed_file"):
                crawler_required = True

        except Exception as error:

            logger.exception("Failed processing record")

            processing_errors.append(str(error))

            results.append({
                "status": "FAILED",
                "error": str(error)
            })

    # Start Glue Crawler once after successful output
    crawler_status = "NOT_REQUIRED"

    if crawler_required:

        try:

            crawler_status = start_glue_crawler()

        except Exception:

            logger.exception("Glue crawler startup failed")

            crawler_status = "START_FAILED"

    logger.info("FINAL RESULTS: %s", results)

    logger.info("CRAWLER STATUS: %s", crawler_status)

    logger.info("========== LAMBDA FINISHED ==========")

    # Report failed records to Lambda
    if processing_errors:

        raise RuntimeError(
            f"ETL failed for {len(processing_errors)} record(s): "
            + "; ".join(processing_errors)
        )

    return {
        "statusCode": 200,
        "results": results,
        "crawler_status": crawler_status
    }
