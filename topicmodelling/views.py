from django.shortcuts import render, redirect
from topicmodelling.models import *
from django.contrib import messages

from topicmodelling.givetopic import *
from topicmodelling.recommend_article import *

def index(request):
    return render(request, 'index.html')

def gettopic(request):
    if request.method=="POST":
        doc=request.POST["doc"]
        topic=answer(doc)
        messages.success(request, topic)
        print(topic)
        # return render(request, "showtopic.html", locals())
    return render(request, 'gettopic.html', locals())

def getarticle(request):
    if request.method=="POST":
        genre=request.POST["genre"]
        titles, articles=closest_doc_name(genre)
        x=zip(titles, articles)
    return render(request, "getarticle.html", locals())
