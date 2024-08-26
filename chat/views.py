import json
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from retrieve import process_row

def chat(request):
    return render(request, 'chat/chat.html')

@csrf_exempt
def send(request):
    data = json.loads(request.body)
    
    row = {'Source': data.get('doc'), 'Source_path': None, 'Question': data.get('question')}
    answer = process_row(row)['Answer']
    response = {"answer": answer}

    return JsonResponse(response)