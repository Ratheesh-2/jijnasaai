from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend.models.schemas import CostSummaryResponse
from backend.services.cost_tracker import CostTracker
from backend.dependencies import get_cost_tracker

router = APIRouter(prefix="/costs", tags=["costs"])


@router.get("/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    conversation_id: Optional[str] = Query(default=None),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
):
    return await cost_tracker.get_cost_summary(conversation_id)
