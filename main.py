from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pickle
import os

app = FastAPI(
    title="PID-5 Clinical Decision Support API",
    description=(
        "Personality disorder screening tool based on PID-5 questionnaire. "
        "Input: 220 item responses (0-3). Output: ICD-10 probability scores. "
        "Reference: Krueger et al. (2012) Psychol Med 42(9); APA DSM-5 AMPD (2013)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
