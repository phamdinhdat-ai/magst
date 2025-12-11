#!/usr/bin/env python3
"""
Scheduled task script for document processing and cleanup.
This script can be run as a cron job to:
1. Process pending documents (move from temp to permanent storage)
2. Clean up expired temporary files
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import argparse

# Add project root to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.session import AsyncSessionLocal
from app.db.models.document import Document, DocumentStatus
from app.crud.crud_document import (
    process_and_move_document, cleanup_temp_files,
    get_documents_by_status
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("document_processor")

async def process_pending_documents(batch_size: int = 100, max_age_minutes: Optional[int] = None):
    """Process pending documents in batches"""
    logger.info("Starting processing of pending documents")
    
    async with AsyncSessionLocal() as db:
        # Get pending documents
        pending_docs = await get_documents_by_status(db, DocumentStatus.PROCESSING, limit=batch_size)
        
        if max_age_minutes is not None:
            # Filter by age if specified
            cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
            pending_docs = [doc for doc in pending_docs if doc.created_at <= cutoff_time]
        
        logger.info(f"Found {len(pending_docs)} pending documents to process")
        
        # Process each document
        success_count = 0
        error_count = 0
        
        for doc in pending_docs:
            try:
                processed_doc = await process_and_move_document(db, doc.id)
                if processed_doc.status == DocumentStatus.COMPLETED:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed to process document {doc.id}: {processed_doc.processing_error}")
            except Exception as e:
                error_count += 1
                logger.exception(f"Error processing document {doc.id}: {str(e)}")
        
        logger.info(f"Processed {success_count} documents successfully, {error_count} failed")
        return success_count, error_count

async def run_cleanup(expiry_hours: int = 24):
    """Clean up expired temporary files"""
    logger.info(f"Starting cleanup of temporary files older than {expiry_hours} hours")
    
    async with AsyncSessionLocal() as db:
        count = await cleanup_temp_files(db)
        logger.info(f"Cleaned up {count} expired temporary files")
        return count

async def run_maintenance(process_docs: bool = True, cleanup: bool = True, batch_size: int = 100, 
                        max_age_minutes: Optional[int] = None, expiry_hours: int = 24):
    """Run document maintenance tasks"""
    logger.info("Starting document maintenance")
    
    results = {
        "processed_count": 0,
        "error_count": 0,
        "cleanup_count": 0
    }
    
    if process_docs:
        success_count, error_count = await process_pending_documents(batch_size, max_age_minutes)
        results["processed_count"] = success_count
        results["error_count"] = error_count
    
    if cleanup:
        cleanup_count = await run_cleanup(expiry_hours)
        results["cleanup_count"] = cleanup_count
    
    logger.info("Document maintenance completed")
    logger.info(f"Results: {results}")
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Document processing and cleanup')
    parser.add_argument('--no-process', action='store_true', help='Skip document processing')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip temporary file cleanup')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-age', type=int, help='Only process documents older than this many minutes')
    parser.add_argument('--expiry-hours', type=int, default=24, help='Cleanup files older than this many hours')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run the maintenance tasks
    asyncio.run(run_maintenance(
        process_docs=not args.no_process,
        cleanup=not args.no_cleanup,
        batch_size=args.batch_size,
        max_age_minutes=args.max_age,
        expiry_hours=args.expiry_hours
    ))
