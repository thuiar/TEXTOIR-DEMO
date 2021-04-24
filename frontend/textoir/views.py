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
    return render(request,'system_information/about_us.html')

@xframe_options_exempt
def copyright(request):
    return render(request,'system_information/copyright.html')

@xframe_options_exempt
def visualization_platform(request):
    return render(request,'system_information/visualization_platform.html')

@xframe_options_exempt
def textoir_toolkit(request):
    return render(request,'system_information/textoir_toolkit.html')

@xframe_options_exempt
def introduction(request):
    return render(request,'system_information/introduction.html')

@xframe_options_exempt
def updates(request):
    return render(request,'system_information/updates.html')

