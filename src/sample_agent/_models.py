"""Models used by the sample agent."""

from pydantic import BaseModel, ConfigDict, Field


class ProductDetails(BaseModel):
    """Describe a product."""

    model_config = ConfigDict(extra="forbid", strict=True)

    product_code: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
