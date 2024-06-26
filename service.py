from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service(resources={"gpu": 1})
class YoloV10:
    def __init__(self):
        from ultralytics import YOLO


        self.model = YOLO("yolov10n.pt")

    @bentoml.api(batchable=True)
    async def predict(self, images: list[Image]) -> list[list[dict]]:
        results = await self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    async def render(self, image: Image) -> Image:
        result = await self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
