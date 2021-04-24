from django.conf.urls import url
from django.urls import path,re_path
from . import views
urlpatterns = [
    url(r'^data_annotation/$',views.data_annotation),
    url(r'^getExampleList/$',views.getExampleList),
    url(r'^updateResultByResultId/$',views.updateResultByResultId),
    url(r'^judgeStateDataAnnotation/$',views.judgeStateDataAnnotation),
    url(r'^judgeStateDataExport2Disk/$',views.judgeStateDataExport2Disk),
    url(r'^getDatasetList/$',views.getDatasetList),
    url(r'^getClassListByDatasetNameAndClassType/$',views.getClassListByDatasetNameAndClassType),
    url(r'^getTextListByDatasetClassTypeLabelName/$',views.getTextListByDatasetClassTypeLabelName),
    url(r'^getTextListByDatasetForUnknown/$',views.getTextListByDatasetForUnknown),
]