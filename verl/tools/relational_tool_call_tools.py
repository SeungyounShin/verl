# examples/tools/relational_all_tools.py
from datetime import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# -----------------------------------------------------------------------------
# In-memory mock data (주신 데이터 그대로)
# -----------------------------------------------------------------------------
USER_DATA = [
    {"id": 1, "name": "Alice", "email": "alice@gmail.com", "location": 1, "favorite_color": "red", "favorite_foods": [1, 2, 3]},
    {"id": 21, "name": "Bob", "email": "bob@hotmail.com", "location": 2, "favorite_color": "orange", "favorite_foods": [4, 5, 6]},
    {"id": 35, "name": "Charlie", "email": "charlie@yahoo.com", "location": 3, "favorite_color": "yellow", "favorite_foods": [3, 7, 2]},
    {"id": 41, "name": "Donna", "email": "donna@example.com", "location": 4, "favorite_color": "green", "favorite_foods": [6, 1, 4]},
    {"id": 42, "name": "Eve", "email": "eve@example.org", "location": 5, "favorite_color": "blue", "favorite_foods": [5, 7, 4]},
    {"id": 43, "name": "Frank The Cat", "email": "frank.the.cat@langchain.dev", "location": 5, "favorite_color": "yellow", "favorite_foods": [3]},
]

LOCATION_DATA = [
    {"id": 1, "city": "New York", "current_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M %p')}", "current_weather": "Partly Cloudy, Temperature: 68°F"},
    {"id": 2, "city": "Los Angeles", "current_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M %p')}", "current_weather": "Sunny, Temperature: 75°F"},
    {"id": 3, "city": "Chicago", "current_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M %p')}", "current_weather": "Mostly Cloudy, Temperature: 60°F"},
    {"id": 4, "city": "Houston", "current_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M %p')}", "current_weather": "Rainy, Temperature: 55°F"},
    {"id": 5, "city": "Miami", "current_time": f"{datetime.now().strftime('%Y-%m-%d %H:%M %p')}", "current_weather": "Partly Cloudy, Temperature: 80°F"},
]

FOOD_DATA = [
    {"id": 1, "name": "Pizza", "calories": 285, "allergic_ingredients": ["Gluten", "Dairy"]},
    {"id": 2, "name": "Chocolate", "calories": 50, "allergic_ingredients": ["Milk", "Soy"]},
    {"id": 3, "name": "Sushi", "calories": 300, "allergic_ingredients": ["Fish", "Soy"]},
    {"id": 4, "name": "Burger", "calories": 350, "allergic_ingredients": ["Gluten", "Dairy"]},
    {"id": 5, "name": "Ice Cream", "calories": 200, "allergic_ingredients": ["Dairy"]},
    {"id": 6, "name": "Pasta", "calories": 180, "allergic_ingredients": ["Gluten"]},
    {"id": 7, "name": "Salad", "calories": 50, "allergic_ingredients": []},
]

CURRENT_USER_ID = 35

_user_by_id = {u["id"]: u for u in USER_DATA}
_loc_by_id = {l["id"]: l for l in LOCATION_DATA}
_food_by_id = {f["id"]: f for f in FOOD_DATA}
_users_by_lcname = {}
for u in USER_DATA:
    _users_by_lcname.setdefault(u["name"].lower(), []).append(u)

def _json_text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)

def _find_users_by_name(name: str) -> List[Dict[str, Any]]:
    q = (name or "").lower().strip()
    res = [{"id": u["id"], "name": u["name"]} for u in _users_by_lcname.get(q, [])]
    if not res:
        for u in USER_DATA:
            if q and q in u["name"].lower():
                res.append({"id": u["id"], "name": u["name"]})
    return res

def _find_locations_by_city(city: str) -> List[Dict[str, Any]]:
    q = (city or "").lower().strip()
    if q in {"la", "l.a."}:
        los = [l for l in LOCATION_DATA if l["city"].lower() == "los angeles"]
        others = sorted([l for l in LOCATION_DATA if l["city"].lower() != "los angeles"], key=lambda x: x["city"])
        return [{"id": l["id"], "city": l["city"]} for l in (los + others)]
    hits = [l for l in LOCATION_DATA if q in l["city"].lower()] if q else []
    if not hits:
        hits = LOCATION_DATA[:]
    return [{"id": l["id"], "city": l["city"]} for l in hits]

def _find_foods_by_name(food: str) -> List[Dict[str, Any]]:
    q = (food or "").lower().strip()
    hits = [f for f in FOOD_DATA if q in f["name"].lower()] if q else []
    if not hits:
        hits = FOOD_DATA[:]
    return [{"id": f["id"], "name": f["name"]} for f in hits]

# -----------------------------------------------------------------------------
# Error handling helpers
# -----------------------------------------------------------------------------
def _safe_int(value, field_name: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None

def _err(msg: str) -> ToolResponse:
    # Align with benchmark.py style: return an error text instead of raising
    return ToolResponse(text=f"Error: {msg}")

def _get_user_or_err(user_id_val) -> tuple[Optional[dict], Optional[ToolResponse]]:
    user_id = _safe_int(user_id_val, "user_id")
    if user_id is None:
        return None, _err("invalid 'user_id'")
    user = _user_by_id.get(user_id)
    if user is None:
        return None, _err(f"user_id {user_id} not found")
    return user, None

def _get_location_or_err(location_id_val) -> tuple[Optional[dict], Optional[ToolResponse]]:
    location_id = _safe_int(location_id_val, "location_id")
    if location_id is None:
        return None, _err("invalid 'location_id'")
    loc = _loc_by_id.get(location_id)
    if loc is None:
        return None, _err(f"location_id {location_id} not found")
    return loc, None

def _get_food_or_err(food_id_val) -> tuple[Optional[dict], Optional[ToolResponse]]:
    food_id = _safe_int(food_id_val, "food_id")
    if food_id is None:
        return None, _err("invalid 'food_id'")
    food = _food_by_id.get(food_id)
    if food is None:
        return None, _err(f"food_id {food_id} not found")
    return food, None

# -----------------------------------------------------------------------------
# Base Tool
# -----------------------------------------------------------------------------
class RelationalBaseTool(BaseTool):
    SCHEMA: OpenAIFunctionToolSchema = None

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        schema = tool_schema or self.SCHEMA
        assert schema is not None, "SCHEMA must be set on subclass"
        super().__init__(config, schema)
        self._instance: Dict[str, Dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance[instance_id] = {"last": None}
        return instance_id, ToolResponse()

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance.pop(instance_id, None)

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        raise NotImplementedError

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
def _schema_of(name: str, desc: str, params: Dict[str, Any]) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
    )

class GetUserNameTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_user_name",
        "Get the name of the user with the given user ID.\n\nArgs:\n  user_id: The user's ID.\n\nReturns:\n  The user's name.",
        {"type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}}, "required": ["user_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        user, err = _get_user_or_err(parameters.get("user_id"))
        if err is not None:
            return err, 0.0, {}
        result = user["name"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class ListUserIdsTool(RelationalBaseTool):
    SCHEMA = _schema_of("list_user_ids", "List all the user IDs.", {"type": "object", "properties": {}, "required": []})
    async def execute(self, instance_id, parameters, **kwargs):
        result = [u["id"] for u in USER_DATA]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class FindUsersByNameTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "find_users_by_name",
        "Find users with the given name.\n\nArgs:\n  name: The name to search for.\n\nReturns:\n  The list of matching users.",
        {"type": "object", "properties": {"name": {"title": "Name", "type": "string"}}, "required": ["name"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        name = str(parameters["name"])
        result = _find_users_by_name(name)
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class FindLocationsByNameTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "find_locations_by_name",
        "Find locations with the given city name.",
        {"type": "object", "properties": {"city": {"title": "City", "type": "string"}}, "required": ["city"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        city = str(parameters["city"])
        result = _find_locations_by_city(city)
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class FindFoodsByNameTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "find_foods_by_name",
        "Find foods with the given name.",
        {"type": "object", "properties": {"food": {"title": "Food", "type": "string"}}, "required": ["food"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        food = str(parameters["food"])
        result = _find_foods_by_name(food)
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class GetUserEmailTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_user_email",
        "Get the email of the user with the given user ID.\n\nArgs:\n  user_id: The user's ID.\n\nReturns:\n  The user's email.",
        {"type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}}, "required": ["user_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        user, err = _get_user_or_err(parameters.get("user_id"))
        if err is not None:
            return err, 0.0, {}
        result = user["email"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetUserLocationTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_user_location",
        "Get the location ID of the user with the given user ID.\n\nArgs:\n  user_id: The user's ID.\n\nReturns:\n  The user's location ID.",
        {"type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}}, "required": ["user_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        user, err = _get_user_or_err(parameters.get("user_id"))
        if err is not None:
            return err, 0.0, {}
        result = user["location"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class GetUserFavoriteColorTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_user_favorite_color",
        "Get the favorite color of the user with the given user ID.\n\nArgs:\n  user_id: The user's ID.\n\nReturns:\n  The user's favorite color.",
        {"type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}}, "required": ["user_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        user, err = _get_user_or_err(parameters.get("user_id"))
        if err is not None:
            return err, 0.0, {}
        result = user["favorite_color"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetUserFavoriteFoodsTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_user_favorite_foods",
        "Get the list of favorite foods of the user with the given user ID.\n\nArgs:\n  user_id: The user's ID.\n\nReturns:\n  The list of favorite foods.",
        {"type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}}, "required": ["user_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        user, err = _get_user_or_err(parameters.get("user_id"))
        if err is not None:
            return err, 0.0, {}
        result = list(user["favorite_foods"])
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class GetWeatherAtLocationTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_weather_at_location",
        "Get the current weather at the location with the given location ID.\n\nArgs:\n  location_id: The location's ID.\n\nReturns:\n  The current weather at the location.",
        {"type": "object", "properties": {"location_id": {"title": "Location Id", "type": "integer"}}, "required": ["location_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        loc, err = _get_location_or_err(parameters.get("location_id"))
        if err is not None:
            return err, 0.0, {}
        result = loc["current_weather"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetCityForLocationTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_city_for_location",
        "Get the city for the location with the given location ID.\n\nArgs:\n  location_id: The location's ID.\n\nReturns:\n  The city name for the location.",
        {"type": "object", "properties": {"location_id": {"title": "Location Id", "type": "integer"}}, "required": ["location_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        loc, err = _get_location_or_err(parameters.get("location_id"))
        if err is not None:
            return err, 0.0, {}
        result = loc["city"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetCurrentTimeForLocationTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_current_time_for_location",
        "Get the current time for the location with the given location ID.\n\nArgs:\n  location_id: The location's ID.\n\nReturns:\n  The current time for the location.",
        {"type": "object", "properties": {"location_id": {"title": "Location Id", "type": "integer"}}, "required": ["location_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        loc, err = _get_location_or_err(parameters.get("location_id"))
        if err is not None:
            return err, 0.0, {}
        result = loc["current_time"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetCurrentWeatherForLocationTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_current_weather_for_location",
        "Get the current weather for the location with the given location ID.\n\nArgs:\n  location_id: The location's ID.\n\nReturns:\n  The current weather for the location.",
        {"type": "object", "properties": {"location_id": {"title": "Location Id", "type": "integer"}}, "required": ["location_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        loc, err = _get_location_or_err(parameters.get("location_id"))
        if err is not None:
            return err, 0.0, {}
        result = loc["current_weather"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetFoodNameTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_food_name",
        "Get the name of the food with the given food ID.\n\nArgs:\n  food_id: The food's ID.\n\nReturns:\n  The name of the food.",
        {"type": "object", "properties": {"food_id": {"title": "Food Id", "type": "integer"}}, "required": ["food_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        food, err = _get_food_or_err(parameters.get("food_id"))
        if err is not None:
            return err, 0.0, {}
        result = food["name"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=result), 0.0, {}

class GetFoodCaloriesTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_food_calories",
        "Get the calories per serving for the food with the given food ID.\n\nArgs:\n  food_id: The food's ID.\n\nReturns:\n  The calories per serving of the food.",
        {"type": "object", "properties": {"food_id": {"title": "Food Id", "type": "integer"}}, "required": ["food_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        food, err = _get_food_or_err(parameters.get("food_id"))
        if err is not None:
            return err, 0.0, {}
        result = food["calories"]
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class GetFoodAllergicIngredientsTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_food_allergic_ingredients",
        "Get the list of allergic ingredients for the food with the given food ID.\n\nArgs:\n  food_id: The food's ID.\n\nReturns:\n  The list of allergic ingredients.",
        {"type": "object", "properties": {"food_id": {"title": "Food Id", "type": "integer"}}, "required": ["food_id"]},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        food, err = _get_food_or_err(parameters.get("food_id"))
        if err is not None:
            return err, 0.0, {}
        result = list(food["allergic_ingredients"])
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

class GetCurrentUserIdTool(RelationalBaseTool):
    SCHEMA = _schema_of(
        "get_current_user_id",
        "Get the current user's ID.\n\nReturns:\n  The current user's ID.",
        {"type": "object", "properties": {}, "required": []},
    )
    async def execute(self, instance_id, parameters, **kwargs):
        result = CURRENT_USER_ID
        self._instance[instance_id]["last"] = result
        return ToolResponse(text=_json_text(result)), 0.0, {}

# -----------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------
def build_relational_tools(config: Optional[dict] = None) -> Dict[str, BaseTool]:
    config = config or {}
    return {
        "get_user_name": GetUserNameTool(config),
        "list_user_ids": ListUserIdsTool(config),
        "find_users_by_name": FindUsersByNameTool(config),
        "find_locations_by_name": FindLocationsByNameTool(config),
        "find_foods_by_name": FindFoodsByNameTool(config),
        "get_user_email": GetUserEmailTool(config),
        "get_user_location": GetUserLocationTool(config),
        "get_user_favorite_color": GetUserFavoriteColorTool(config),
        "get_user_favorite_foods": GetUserFavoriteFoodsTool(config),
        "get_weather_at_location": GetWeatherAtLocationTool(config),
        "get_city_for_location": GetCityForLocationTool(config),
        "get_current_time_for_location": GetCurrentTimeForLocationTool(config),
        "get_current_weather_for_location": GetCurrentWeatherForLocationTool(config),
        "get_food_name": GetFoodNameTool(config),
        "get_food_calories": GetFoodCaloriesTool(config),
        "get_food_allergic_ingredients": GetFoodAllergicIngredientsTool(config),
        "get_current_user_id": GetCurrentUserIdTool(config),
    }

def get_openai_tools_schema_list() -> List[dict]:
    """tools list"""
    return [
        GetUserNameTool.SCHEMA.model_dump(),
        ListUserIdsTool.SCHEMA.model_dump(),
        FindUsersByNameTool.SCHEMA.model_dump(),
        FindLocationsByNameTool.SCHEMA.model_dump(),
        FindFoodsByNameTool.SCHEMA.model_dump(),
        GetUserEmailTool.SCHEMA.model_dump(),
        GetUserLocationTool.SCHEMA.model_dump(),
        GetUserFavoriteColorTool.SCHEMA.model_dump(),
        GetUserFavoriteFoodsTool.SCHEMA.model_dump(),
        GetWeatherAtLocationTool.SCHEMA.model_dump(),
        GetCityForLocationTool.SCHEMA.model_dump(),
        GetCurrentTimeForLocationTool.SCHEMA.model_dump(),
        GetCurrentWeatherForLocationTool.SCHEMA.model_dump(),
        GetFoodNameTool.SCHEMA.model_dump(),
        GetFoodCaloriesTool.SCHEMA.model_dump(),
        GetFoodAllergicIngredientsTool.SCHEMA.model_dump(),
        GetCurrentUserIdTool.SCHEMA.model_dump(),
    ]
