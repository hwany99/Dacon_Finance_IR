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

def get_LLM_output(prompt):
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
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0]["generated_text"][len(LLM_input):]
    return response

def get_QA_output(context, question):
    prompt = f"""
    본문:
    {context}

    질문: {question}
    
    챗봇으로서 주어진 질문에만 완전한 문장으로 답변해주세요. 금액은 단위를 포함하여 단답형으로 답변해주세요.
    답변:
    """

    output = get_LLM_output(prompt)
    return output

def get_Rerank_output(results, question):
    context = []
    full_doc_text = ""
    for i in range(0, len(results)):
        if i > 4: break
        context.append(results[i].page_content)
        text = f"[문서 {i+1}]" + results[i].page_content
        full_doc_text += f"\n\n{text}\n==="

    prompt = f"""
    정보: 
        {full_doc_text}

    위 정보에서 아래 질문에 대한 답을 얻을 수 있는 문서를 2개 골라 번호를 출력하세요.

    질문: {question}

    문서 번호: """

    output = get_LLM_output(prompt)

    reranked_idx_list = []
    for char in output.split('\n')[0]:
        if char.isdigit() and char != "0":
            idx = int(char)
            if idx-1 not in reranked_idx_list and 1 <= idx <= 5:
                reranked_idx_list.append(idx-1)
    
    if len(reranked_idx_list) < 2 and 0 not in reranked_idx_list:
        reranked_idx_list.append(0)
    
    final_context = ""
    for idx in reranked_idx_list:
        final_context += f"\n{results[idx].page_content}\n"
    print(reranked_idx_list, output, '\n--------')
    # print('\n'.join([r.page_content for r in results]))
    # print('-----------------')
    # print(final_context)
    
    return final_context, full_doc_text