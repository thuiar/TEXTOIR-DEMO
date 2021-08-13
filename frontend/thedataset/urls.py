"""the dataset  URL Configuration

"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path

from . import views
urlpatterns = [
    path('toDatasetList/',views.toDatasetList),
    path('getDatasetList/',views.getDatasetList),
    path('toAddHtml/',views.toAddHtml),
    url(r'^addDataset/$',views.addDataset),
    url(r'^details/$',views.details),
    url(r'^downloadDataset/$',views.downloadDataset),
    url(r'^toEdit/$', views.toEdit),
    url(r'^editData/$', views.editData),
    url(r'^delData/$', views.delData),
    url(r'^update_source/$', views.update_source),

    
]
