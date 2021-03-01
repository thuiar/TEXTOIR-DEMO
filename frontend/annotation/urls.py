from django.conf.urls import url
from django.urls import path,re_path
from . import views
urlpatterns = [
    url(r'^data_annotation/$',views.data_annotation),
    url(r'^getExampleList/$',views.getExampleList),
    url(r'^updateResultByResultId/$',views.updateResultByResultId),
]