
from pydantic import BaseModel, Field
from enum import Enum
import datetime
from typing import Optional, List, Dict, Any


class EquipmentTypeStruct(BaseModel):
    """
    Container type classification hierarchy.
    """
    code: Optional[str] = Field(None, description='Primary equipment type code.')
    parent_code_1: Optional[str] = Field(None, description='')
    parent_code_2: Optional[str] = Field(None, description='')



class OwnershipStruct(BaseModel):
    """
    Ownership and leasing details for the container.
    """
    ownership_type_code: Optional[str] = Field(None, description='')
    ownership_type_name: Optional[str] = Field(None, description='')
    ownership_contract_number: Optional[str] = Field(None, description='')
    leasing_contract_number: Optional[str] = Field(None, description='')
    leasing_company_code: Optional[str] = Field(None, description='')



class LifecycleStruct(BaseModel):
    """
    Manufacturing and lifecycle-related timestamps.
    """
    infleet_datetime: Optional[datetime.datetime] = Field(None, description='')
    infleet_location: Optional[str] = Field(None, description='')
    manufacturing_date: Optional[datetime.datetime] = Field(None, description='')
    refurbishment_date: Optional[datetime.datetime] = Field(None, description='')



class InspectionsStruct(BaseModel):
    """
    CSC/PMI inspection details.
    """
    csc_inspection_date: Optional[datetime.datetime] = Field(None, description='')
    csc_inspection_location: Optional[str] = Field(None, description='')
    csc_inspection_next_date: Optional[datetime.datetime] = Field(None, description='')
    pmi_inspection_date: Optional[datetime.datetime] = Field(None, description='')
    pmi_inspection_location: Optional[str] = Field(None, description='')
    pmi_inspection_next_date: Optional[datetime.datetime] = Field(None, description='')
    inspector: Optional[str] = Field(None, description='')
    inspector_location: Optional[str] = Field(None, description='')



class ModificationStruct(BaseModel):
    """
    Container modification or repair program information.
    """
    program: Optional[str] = Field(None, description='')
    status: Optional[str] = Field(None, description='')



class ManufacturerInfoStruct(BaseModel):
    """
    Information about the container manufacturer and production.
    """
    manufacturer_code: Optional[str] = Field(None, description='')
    manufacturer_number: Optional[str] = Field(None, description='')
    production_serial_number: Optional[str] = Field(None, description='')
    construction_material_code: Optional[str] = Field(None, description='')
    equipment_profile_name: Optional[str] = Field(None, description='')
    equipment_profile_description: Optional[str] = Field(None, description='')
    equipment_profile_comment: Optional[str] = Field(None, description='')



class WeightStruct(BaseModel):
    """
    Container weight specifications.
    """
    tare_weight: Optional[float] = Field(None, description='')
    tare_weight_unit: Optional[str] = Field(None, description='')
    maximum_payload_weight: Optional[float] = Field(None, description='')
    maximum_payload_weight_unit: Optional[str] = Field(None, description='')
    maximum_gross_weight: Optional[float] = Field(None, description='')



class Container(BaseModel):
    """
    A physical shipping container with structural, ownership, and operational attributes.
    """
    container_id: str = Field(..., description='Container identifier, derived from equipment_number.')
    sequence_number: Optional[int] = Field(None, description='Internal sequence number associated with the equipment record.')
    is_active: Optional[bool] = Field(None, description='Whether the container is currently active in operations.')
    equipment_size: Optional[str] = Field(None, description='Container size designation (e.g., 20GP, 40GP, 40HC).')
    equipment_type: Optional[EquipmentTypeStruct] = Field(None, description='Container type classification hierarchy.')
    ownership: Optional[OwnershipStruct] = Field(None, description='Ownership and leasing details for the container.')
    lifecycle: Optional[LifecycleStruct] = Field(None, description='Manufacturing and lifecycle-related timestamps.')
    inspections: Optional[InspectionsStruct] = Field(None, description='CSC/PMI inspection details.')
    modification: Optional[ModificationStruct] = Field(None, description='Container modification or repair program information.')
    manufacturer_info: Optional[ManufacturerInfoStruct] = Field(None, description='Information about the container manufacturer and production.')
    weight: Optional[WeightStruct] = Field(None, description='Container weight specifications.')
    last_event_timestamp: Optional[datetime.datetime] = Field(None, description='Last known operational event timestamp for the container.')

