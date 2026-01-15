from pydantic import BaseModel, Field
import datetime
from typing import Optional, List


class BookedByStruct(BaseModel):
    """
    Basic commercial booking party information.
    """

    code: Optional[str] = Field(None, description="")
    name: Optional[str] = Field(None, description="")
    office: Optional[str] = Field(None, description="")


class ConsigneeStruct(BaseModel):
    """
    Consignee customer details.
    """

    code: Optional[str] = Field(None, description="")
    name: Optional[str] = Field(None, description="")


class ShipperStruct(BaseModel):
    """
    Shipper customer details.
    """

    customer_code: Optional[str] = Field(None, description="")
    customer_id: Optional[str] = Field(None, description="")


class ContainersItemStruct(BaseModel):
    """
    Item struct for ContainersItemStruct
    """

    container_number: Optional[str] = Field(None, description="")
    container_size: Optional[int] = Field(None, description="")
    container_type: Optional[str] = Field(None, description="")
    weight_kg: Optional[float] = Field(None, description="")
    teu: Optional[float] = Field(None, description="")


class TransportPlanStruct(BaseModel):
    """
    Simplified current transport plan details.
    """

    planned_load_terminal_code: Optional[str] = Field(None, description="")
    planned_vessel_code: Optional[str] = Field(None, description="")
    planned_voyage_number: Optional[str] = Field(None, description="")
    estimated_final_arrival: Optional[datetime.datetime] = Field(None, description="")


class LegsAfterLastDischargeItemStruct(BaseModel):
    """
    Item struct for LegsAfterLastDischargeItemStruct
    """

    start_city: Optional[str] = Field(None, description="")
    end_city: Optional[str] = Field(None, description="")
    vessel_code: Optional[str] = Field(None, description="")
    service_code: Optional[str] = Field(None, description="")
    transport_mode: Optional[str] = Field(None, description="")
    arrival_time: Optional[datetime.datetime] = Field(None, description="")
    departure_time: Optional[datetime.datetime] = Field(None, description="")


class LegsBeforeFirstLoadItemStruct(BaseModel):
    """
    Item struct for LegsBeforeFirstLoadItemStruct
    """

    start_city: Optional[str] = Field(None, description="")
    end_city: Optional[str] = Field(None, description="")
    vessel_code: Optional[str] = Field(None, description="")
    service_code: Optional[str] = Field(None, description="")
    transport_mode: Optional[str] = Field(None, description="")
    arrival_time: Optional[datetime.datetime] = Field(None, description="")
    departure_time: Optional[datetime.datetime] = Field(None, description="")


class VasItemStruct(BaseModel):
    """
    Item struct for VasItemStruct
    """

    vas_code: Optional[str] = Field(None, description="")
    vas_description: Optional[str] = Field(None, description="")


class Shipment(BaseModel):
    """
    A shipment or booking representing the transport of one or more containers.
    """

    shipment_number: str = Field(
        ..., description="Unique shipment or booking identifier."
    )
    number_of_containers: Optional[int] = Field(
        None, description="Total number of containers in the shipment."
    )
    total_teu: Optional[float] = Field(None, description="Total TEU for the shipment.")
    total_weight_kg: Optional[float] = Field(
        None, description="Total weight in kilograms."
    )
    is_delivered: Optional[bool] = Field(
        None, description="Whether the shipment has been delivered."
    )
    is_loaded: Optional[bool] = Field(
        None, description="Whether shipment containers have been loaded on a vessel."
    )
    is_pending: Optional[bool] = Field(
        None, description="Whether the shipment is still pending operational actions."
    )
    place_of_receipt_city: Optional[str] = Field(
        None, description="City where shipment is first received."
    )
    place_of_delivery_city: Optional[str] = Field(
        None, description="Final delivery city of the shipment."
    )
    last_eta: Optional[datetime.datetime] = Field(
        None,
        description="Latest known estimated time of arrival (derived from customer promised ETA).",
    )
    latest_schedule_update_time: Optional[datetime.datetime] = Field(
        None, description="When the schedule was last updated."
    )
    booked_by: Optional[BookedByStruct] = Field(
        None, description="Basic commercial booking party information."
    )
    consignee: Optional[ConsigneeStruct] = Field(
        None, description="Consignee customer details."
    )
    shipper: Optional[ShipperStruct] = Field(
        None, description="Shipper customer details."
    )
    containers: Optional[List[ContainersItemStruct]] = Field(
        None, description="List of containers associated with this shipment."
    )
    transport_plan: Optional[TransportPlanStruct] = Field(
        None, description="Simplified current transport plan details."
    )
    legs_after_last_discharge: Optional[List[LegsAfterLastDischargeItemStruct]] = Field(
        None, description="Operational legs after last discharge (GCSS RKEM)."
    )
    legs_before_first_load: Optional[List[LegsBeforeFirstLoadItemStruct]] = Field(
        None, description="Pre-carriage or pre-load legs."
    )
    vas: Optional[List[VasItemStruct]] = Field(None, description="")
