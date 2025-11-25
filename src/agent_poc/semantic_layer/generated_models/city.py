
from pydantic import BaseModel, Field
from enum import Enum
import datetime
from typing import Optional, List, Dict, Any


class City(BaseModel):
    city: Optional[str] = Field(None, description='City where the facility is located.')
    country_code: Optional[str] = Field(None, description='Country code of the facility.')
    country_name: Optional[str] = Field(None, description='Country name of the facility.')

