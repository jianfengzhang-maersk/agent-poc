
from pydantic import BaseModel, Field
from enum import Enum
import datetime
from typing import Optional, List, Dict, Any


class GeoPositionStruct(BaseModel):
    latitude: Optional[float] = Field(None, description='')
    longitude: Optional[float] = Field(None, description='')



class FacilityTypesItemStruct(BaseModel):
    type_code: Optional[str] = Field(None, description='')
    type_name: Optional[str] = Field(None, description='')



class DefinedAreasItemStruct(BaseModel):
    area_name: Optional[str] = Field(None, description='')
    area_type_code: Optional[str] = Field(None, description='')
    location_type: Optional[str] = Field(None, description='')



class FacilityAvailabilitiesItemStruct(BaseModel):
    start_time: Optional[str] = Field(None, description='')
    end_time: Optional[str] = Field(None, description='')
    availability_type: Optional[str] = Field(None, description='')
    weekdays: Optional[Dict[str, Any]] = Field(None, description='Key: weekday, Value: open/close indicator')



class FacilityOfferingsItemStruct(BaseModel):
    offering_code: Optional[str] = Field(None, description='')
    offering_name: Optional[str] = Field(None, description='')
    offering_description: Optional[str] = Field(None, description='')
    valid_to: Optional[datetime.datetime] = Field(None, description='')
    is_primary: Optional[bool] = Field(None, description='')



class Facility(BaseModel):
    facility_id: str = Field(..., description='Unique identifier of the facility, derived from facilityIdentifier.')
    facility_name: Optional[str] = Field(None, description='Human-readable name of the facility.')
    is_active: Optional[bool] = Field(None, description='Whether the facility is operational.')
    city: Optional[str] = Field(None, description='City where the facility is located.')
    postal_code: Optional[str] = Field(None, description='Postal or ZIP code of the facility.')
    geo_position: Optional[GeoPositionStruct] = Field(None, description='Geographic coordinates of the facility.')
    facility_types: Optional[List[FacilityTypesItemStruct]] = Field(None, description='List of facility type classifications (e.g., terminal, warehouse, depot, customer location).')
    defined_areas: Optional[List[DefinedAreasItemStruct]] = Field(None, description='Sub-areas within the facility, such as gates, yards, berths.')
    facility_availabilities: Optional[List[FacilityAvailabilitiesItemStruct]] = Field(None, description='Operating time windows when the facility is available.')
    facility_offerings: Optional[List[FacilityOfferingsItemStruct]] = Field(None, description='Services the facility provides (e.g., lift-on, storage, reefer plug).')
    iata_code: Optional[str] = Field(None, description='IATA airport code (if applicable).')
    geoid: Optional[str] = Field(None, description='Geographical ID for location matching.')
    hsud_code: Optional[str] = Field(None, description='HSUD code used in operational systems.')
    smdg: Optional[str] = Field(None, description='SMDG terminal code (if applicable).')

