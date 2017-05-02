from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from .test import return_titles
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.utils.safestring import mark_safe

# Create your views here.
def index(request):
    output_list = ''
    output=''
    ipt =''
    books = return_titles(True)
    #authors = return_titles(False)
    if request.GET.get('search'):
        search = request.GET.get('search')
        ipt, output_list = find_similar(search)#, True)
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
    return render_to_response('project_template/index.html', 
                          {'output': output,
                           'magic_url': request.get_full_path(),
                           'input': ipt,
                           'books': mark_safe(books)
                           #'authors': mark_safe(authors)
                           })