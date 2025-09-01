"""
Database factory for G-Eval FastAPI application.

This module provides a factory pattern to switch between SQLite and DynamoDB
database managers based on configuration.
"""

from typing import Union
from settings import config
from database import DatabaseManager
from dynamodb_manager import DynamoDBManager


def get_database_manager() -> Union[DatabaseManager, DynamoDBManager]:
    """
    Factory function to get the appropriate database manager based on configuration.
    
    Returns:
        DatabaseManager or DynamoDBManager instance based on DATABASE_TYPE setting
    """
    if config.database_type.lower() == 'dynamodb':
        return DynamoDBManager(
            region=config.aws_region,
            endpoint_url=config.dynamodb_endpoint
        )
    else:
        return DatabaseManager()


# Global database manager instance
db_manager = get_database_manager()
