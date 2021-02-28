"""the dataset  URL Configuration

"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('model_management/',views.model_management),
    path('model_management/getModelList',views.getModelList),
    url(r'^model_management/details/$',views.model_management_details),

    
    path('model_training/',views.model_training),
    path('model_training/getModelList',views.getModelLogList),
    url(r'^model_training/toLogParameter/$',views.toLogParameter),

    path('model_training/toRunModel',views.toRunModel),
    path('model_training/getParamListByModelId',views.getParamListByModelId),
    path('model_training/add_model_training_log',views.add_model_training_log),

    
    path('model_test/',views.model_test),
    
    path('model_analysis/',views.model_analysis),
    path('model_analysis/getModelAnalysisExampleData',views.getModelAnalysisExampleData),
    path('model_analysis/modelAnalysisTest',views.modelAnalysisTest),
    
]

























