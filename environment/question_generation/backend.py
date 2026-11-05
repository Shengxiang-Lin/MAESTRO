"""Question-author and semantic-critic backends."""
from __future__ import annotations
import json,os,urllib.error,urllib.request
from dataclasses import dataclass
from typing import Any,Mapping
class BackendError(RuntimeError):pass

def _extract_content(payload:Mapping[str,Any])->str:
    choices=payload.get("choices")
    if isinstance(choices,list) and choices:
        message=choices[0].get("message",{}) if isinstance(choices[0],Mapping) else {}
        content=message.get("content") if isinstance(message,Mapping) else None
        if isinstance(content,str):return content
    output=payload.get("output")
    if isinstance(output,list):
        texts=[]
        for item in output:
            if not isinstance(item,Mapping):continue
            for content in item.get("content",()) if isinstance(item.get("content"),list) else ():
                if isinstance(content,Mapping) and isinstance(content.get("text"),str):texts.append(content["text"])
        if texts:return "\n".join(texts)
    raise BackendError("OpenAI-compatible response did not contain assistant text")
def _json_object(text:str)->dict[str,Any]:
    stripped=text.strip()
    if stripped.startswith("```"):
        stripped=stripped.strip("`").strip()
        if stripped.lower().startswith("json"):stripped=stripped[4:].strip()
    try:payload=json.loads(stripped)
    except json.JSONDecodeError:
        start=stripped.find("{");end=stripped.rfind("}")
        if start<0 or end<=start:raise BackendError("model did not return a JSON object")
        try:payload=json.loads(stripped[start:end+1])
        except json.JSONDecodeError as exc:raise BackendError("model returned invalid JSON") from exc
    if not isinstance(payload,dict):raise BackendError("model returned non-object JSON")
    return payload
@dataclass
class OpenAICompatibleBackend:
    base_url:str;model:str;api_key:str|None=None;temperature:float=.4;max_tokens:int=2048;timeout:float=180.;json_mode:bool=True
    def complete_json(
        self,system_prompt:str,user_prompt:str,*,
        response_schema:Mapping[str,Any]|None=None,
    )->dict[str,Any]:
        """JSON mode by default; optional schema-constrained JSON for vLLM.

        An explicit unsupported-schema HTTP 400/422 may downgrade this *one*
        request to JSON object mode. Other errors are not silently retried.
        The caller retains its own deterministic and semantic validators.
        """
        url=self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):url += "/chat/completions"
        body={"model":self.model,
              "messages":[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
              "temperature":self.temperature,"max_tokens":self.max_tokens}
        if response_schema is not None and self.json_mode:
            body["response_format"]={
                "type":"json_schema",
                "json_schema":{"name":"subagent_feedback","strict":True,
                               "schema":response_schema},
            }
        elif self.json_mode:
            body["response_format"]={"type":"json_object"}
        headers={"Content-Type":"application/json"}
        if self.api_key:headers["Authorization"]="Bearer "+self.api_key

        def invoke(request_body):
            request=urllib.request.Request(
                url,data=json.dumps(request_body,ensure_ascii=False).encode("utf-8"),
                headers=headers,method="POST",
            )
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            payload=invoke(body)
        except urllib.error.HTTPError as exc:
            # Some older OpenAI-compatible servers implement json_object but
            # not json_schema. Fall back ONLY when the API explicitly reports
            # that json_schema/response_format is unsupported.
            details=exc.read().decode("utf-8","replace")[:1024]
            schema_unsupported=(
                response_schema is not None and self.json_mode
                and exc.code in (400,422)
                and ("json_schema" in details.lower()
                     or "response_format" in details.lower())
                and any(term in details.lower() for term in (
                    "not support","unsupported","invalid","unknown",
                    "unrecognized","not implemented",
                ))
            )
            if not schema_unsupported:
                raise BackendError(
                    f"OpenAI-compatible call failed: HTTP {exc.code}: {details[:180]}"
                ) from exc
            body["response_format"]={"type":"json_object"}
            try:
                payload=invoke(body)
            except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as retry_exc:
                raise BackendError(
                    f"OpenAI-compatible schema fallback failed: {retry_exc}"
                ) from retry_exc
        except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as exc:
            raise BackendError(f"OpenAI-compatible call failed: {exc}") from exc
        try:
            return _json_object(_extract_content(payload))
        except BackendError as exc:
            # Attach per-call diagnostics, never shared mutable client state.
            choices=payload.get("choices") or []
            first=choices[0] if choices and isinstance(choices[0],dict) else {}
            message=first.get("message") or {}
            exc.diagnostic={
                "finish_reason":first.get("finish_reason"),
                "usage":payload.get("usage"),
                "content":message.get("content") if isinstance(message,dict) else None,
            }
            raise

class PassthroughAuthorBackend:
    def author(self,payload:Mapping[str,Any],prompt:str)->dict[str,Any]:return {"question":str(payload.get("draft_question") or "").strip(),"notes":"passthrough"}
class LLMQuestionAuthor:
    def __init__(self,backend:OpenAICompatibleBackend):self.backend=backend
    def author(self,payload,prompt):return self.backend.complete_json("You are a benchmark question author. Follow the public frozen contract exactly. Return JSON only.",prompt)
class PassthroughCriticBackend:
    def review(self,payload,question,prompt):return {"accept":True,"semantic_fidelity":"pass","repair_question":question,"issues":[]}
class LLMSemanticCritic:
    def __init__(self,backend:OpenAICompatibleBackend):self.backend=backend
    def review(self,payload,question,prompt):return self.backend.complete_json("You are a strict semantic fidelity critic for a frozen benchmark task. Return JSON only.",prompt)
__all__=["BackendError","LLMQuestionAuthor","LLMSemanticCritic","OpenAICompatibleBackend","PassthroughAuthorBackend","PassthroughCriticBackend"]
