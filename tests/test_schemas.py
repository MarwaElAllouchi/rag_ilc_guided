from app.api.schemas import ChatRequest, ChatResponse


request = ChatRequest(question="Comment s'inscrire à l'école ?", top_k=5)
response = ChatResponse(answer="Vous pouvez vous inscrire en ligne ou au secrétariat.")

print(request.model_dump())
print(response.model_dump())