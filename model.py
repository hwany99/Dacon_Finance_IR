import torch
import transformers

def get_LLM_pipeline():
    model_id = "rtzr/ko-gemma-2-9b-it"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.model.eval()
    return pipeline

LLM_pipeline = get_LLM_pipeline()
terminators = [
    LLM_pipeline.tokenizer.eos_token_id,
    LLM_pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
]

def get_LLM_output(prompt, temperature=0.6):
    global LLM_pipeline, terminators

    messages = [
        {"role": "user", "content": prompt}
    ]

    LLM_input = LLM_pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    outputs = LLM_pipeline(
        LLM_input,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

    response = outputs[0]["generated_text"][len(LLM_input):]
    return response

def get_QA_output(context, question):
    prompt = f"""
    다음 정보를 바탕으로 질문에 답하세요:
    {context}

    질문: {question}
 
    주어진 질문에만 완전한 문장으로 답변해주세요. 답변할 때 질문의 주어를 써주세요.
    
    근거 및 추진에 대한 질문은 사업근거 및 추진경위 문장을 참고하세요.
    금액에 대한 질문은 단위를 포함하여 단답형으로 답변해주세요.
    
    답변:
    """

    output = get_LLM_output(prompt, 0.6)
    return output

# return: final_context, full_text
# final_context: 최종 llm에게 넘길 context
# full_doc_text: reranking하기 전 context
def get_Rerank_output(documents, question):
    context = []
    full_doc_text = ""
    for i in range(len(documents)):
        # 5개만 뽑음
        if i > 4:
            break
        # context에 0,1,2,3,4 까지 포함 
        context.append(documents[i].page_content)
        # index가 1,2,3,4인 경우만 rerank 진행
        if i != 0:
            text = f"문서{i}:" + "\n\n" + documents[i].page_content
            full_doc_text += f"\n\n{text}\n==="
    full_doc_text.replace("None"," ")
    prompt = f"""
    정보: 
        {full_doc_text}

    위 정보에서 아래 질문에 대한 텍스트를 포함하는지 확인하고 답을 유추할 수 있고 가장 유사한 문서를 1~2개만 골라서 문서의 번호만 출력해줘
    만약 텍스트가 길다면 1개만 출력해주고
    유사한 문서가 없다면 ""를 출력해줘 
    
    질문: {question}

    """

    output = get_LLM_output(prompt)
    #print(f"Answer: 0, {output}\n") -> debuging용
    

    reranked_idx_list = [0]
    
    # 1,2,3,4중 reranking된 index 추출
    for char in output:
        if char.isdigit() and char != "0":
            idx = int(char)
            if idx not in reranked_idx_list and idx < 5:
                reranked_idx_list.append(idx)
                
    #print(f"reranked_list: {reranked_idx_list}\n") -> debuging용
    final_context = ""
    for idx in reranked_idx_list:
        final_context += f"\n{context[idx]}\n"
    return final_context, full_doc_text