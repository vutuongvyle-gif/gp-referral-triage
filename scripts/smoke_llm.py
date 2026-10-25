from pydantic import BaseModel, Field
from referral_triage.llm import get_llm, get_embeddings


class Check(BaseModel):
    specialty: str = Field(description="Most likely specialty for the referral")
    urgent: bool


llm = get_llm().with_structured_output(Check)
print(llm.invoke("62-year-old with 8 months of knee pain, worse on stairs. Which specialty?"))
print(len(get_embeddings().embed_query("knee osteoarthritis")), "dimensions")