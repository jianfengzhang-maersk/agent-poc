from pydantic import BaseModel, Field
from typing import Optional


class City(BaseModel):
    """
    A geographical city where logistics operations such as container movements, terminals, and facilities exist.
    """

    city: Optional[str] = Field(None, description="City where the facility is located.")
    country_code: Optional[str] = Field(
        None, description="Country code of the facility."
    )
    country_name: Optional[str] = Field(
        None, description="Country name of the facility."
    )
