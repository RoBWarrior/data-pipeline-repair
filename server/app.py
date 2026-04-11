import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import PipelineAction, PipelineObservation, PipelineState
from server.environment import DataPipelineEnvironment
from server.tasks import EASY_TASK_ID, MEDIUM_TASK_ID, HARD_TASK_ID

app = FastAPI(
    title="Data Pipeline Repair Environment",
    description="An OpenEnv environment where agents learn to fix broken data pipelines.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = DataPipelineEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = EASY_TASK_ID


class StepRequest(BaseModel):
    command: str
    parameters: dict = {}


class StepResponse(BaseModel):
    observation: PipelineObservation
    reward: float
    done: bool
    info: dict = {}


# ------------------------------------------------------------------ #
#  ENDPOINTS                                                           #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok", "environment": "data-pipeline-repair"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    valid_tasks = [EASY_TASK_ID, MEDIUM_TASK_ID, HARD_TASK_ID]
    task_id = request.task_id if request.task_id in valid_tasks else EASY_TASK_ID
    obs = env.reset(task_id=task_id)
    return obs


@app.post("/step")
def step(request: StepRequest) -> StepResponse:
    action = PipelineAction(
        command=request.command,
        parameters=request.parameters
    )
    obs, reward, done = env.step(action)
    
    # ← KEY FIX: always done=True like DragonEye
    return StepResponse(
        observation=obs,
        reward=reward,
        done=True,  # ← every step is terminal
        info={"step": obs.step_number, "score": obs.score_so_far}
    )


@app.get("/state")
def state() -> PipelineState:
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": EASY_TASK_ID,
                "difficulty": "easy",
                "description": "Fix column types and fill nulls in employee dataset"
            },
            {
                "id": MEDIUM_TASK_ID,
                "difficulty": "medium",
                "description": "Fix schema drift, duplicates, date formats in sales dataset"
            },
            {
                "id": HARD_TASK_ID,
                "difficulty": "hard",
                "description": "Fix and join two broken tables with multiple issues"
            }
        ]
    }
    
    
    
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()