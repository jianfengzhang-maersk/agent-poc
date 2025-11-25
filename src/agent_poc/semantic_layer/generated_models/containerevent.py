
from pydantic import BaseModel, Field
from enum import Enum
import datetime
from typing import Optional, List, Dict, Any


class GeoPositionStruct(BaseModel):
    latitude: Optional[float] = Field(None, description='')
    longitude: Optional[float] = Field(None, description='')



class Containerevent(BaseModel):
    event_type: Optional[str] = Field(None, description='The type of container movement event, derived from move_type (e.g., gate_in, gate_out, load, discharge).')
    event_time: Optional[datetime.datetime] = Field(None, description='The timestamp when the event occurred, derived from activity_time.')
    container_id: Optional[str] = Field(None, description='Container number, derived from equipment_number.')
    location_code: Optional[str] = Field(None, description='Code representing where the event occurred, derived from event_location_code.')
    is_empty: Optional[bool] = Field(None, description='Whether the container was empty during the event.')
    transport_mode: Optional[str] = Field(None, description='Transport mode associated with the event.')
    event_reason: Optional[str] = Field(None, description='Additional classification of the event reason.')
    geo_position: Optional[GeoPositionStruct] = Field(None, description='Latitude and longitude of the event location.')
    voyage_number: Optional[str] = Field(None, description='Voyage number associated with this movement.')
    vessel_code: Optional[str] = Field(None, description='Maersk vessel code associated with the event, derived from vessel_maersk_code.')
    shipment_number: Optional[str] = Field(None, description='Shipment number associated with this event.')
    freight_order: Optional[str] = Field(None, description='Freight order identifier associated with the event.')
    bl_number: Optional[str] = Field(None, description='Bill of lading number related to this event.')

