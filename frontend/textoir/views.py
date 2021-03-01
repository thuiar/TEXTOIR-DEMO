from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt

# Create your views here.

def index(request):
    return render(request,'index.html')

@xframe_options_exempt
def welcome(request):
    return render(request,'home-page.html')
    
@xframe_options_exempt
def about_us(request):
    return render(request,'about_us.html')