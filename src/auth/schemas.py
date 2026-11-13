"""즐겨찾기 API 입출력 스키마."""

from pydantic import BaseModel, Field

MAX_FAVORITES = 100


class FavoriteIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    mobile: str = Field(default="", max_length=20)
    city: str = Field(default="", max_length=20)
    next: str = Field(default="", max_length=80)


class FavoriteItem(FavoriteIn):
    id: str = Field(pattern=r"^\d{9}$")


class MergeRequest(BaseModel):
    items: list[FavoriteItem] = Field(default=[], max_length=MAX_FAVORITES)
