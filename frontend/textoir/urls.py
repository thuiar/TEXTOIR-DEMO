"""textoir URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from django.urls import path
from .views import index,welcome,about_us,introduction,updates,copyright,visualization_platform,textoir_toolkit
urlpatterns = [
    path('', index),
    url(r'^admin/', admin.site.urls),
    path('welcome/',welcome),
    path('about_us/',about_us),
    path('copyright/',copyright),
    path('introduction/',introduction),
    path('visualization_platform/',visualization_platform),
    path('textoir_toolkit/',textoir_toolkit),
    path('updates/',updates),

    # include the dataset ' urls
    url(r'^thedataset/',include('thedataset.urls')),
    # include detection ' urls
    url(r'^detection/',include('detection.urls')),
    # include discovery ' urls
    url(r'^discovery/',include('discovery.urls')),
    # include annotation ' urls
    url(r'^annotation/',include('annotation.urls')),
]
