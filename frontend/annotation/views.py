from django.shortcuts import render
from thedataset import models
from django.http import JsonResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
# 分页
from django.core.paginator import Paginator
# 查询数据库结果转字典。
from django.forms.models import model_to_dict
from django.core import serializers
# Create your views here.
@xframe_options_exempt
def data_annotation(request):#
    
    dataset_name = request.GET.get('dataset_name')
    model_detection = request.GET.get("model_detection")
    
    dataset_List = models.DataSet.objects.values()
    modelList_detection = models.Model_Tdes.objects.values().filter(type = 1)
    # example_list = models.Model_Test_Example.objects.values().filter(type="12")

    result = {}
    result['dataset_name'] = dataset_name
    result['dataset_List'] = dataset_List
    result['model_detection'] = model_detection
    result['modelList_detection'] = modelList_detection
    # result['exampleList'] = example_list

    return render(request,'annotation/data_annotation.html',result)


@csrf_exempt
def getExampleList(request):
    # model_id = request.GET.get('model_id')
    dataset_id = request.GET.get('dataset_name')
    model_detection = request.GET.get("model_detection")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    if dataset_id=='null':
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    # get Example list
    exampleList = models.Data_Note_Annotation_Result.objects.values().filter(dataset_id=dataset_id).order_by('result_id')
    count = exampleList.count()
    # 分页
    paginator = Paginator(exampleList, limit)
    exampleList = paginator.get_page(page)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(exampleList)
    
    return JsonResponse(result)

@csrf_exempt
def updateResultByResultId(request):
    # model_id = request.GET.get('model_id')
    
    result_id = request.POST['result_id']
    val = request.POST['val']

    try:
        # ## save msg to Database
        record = models.Data_Note_Annotation_Result.objects.get(result_id=result_id)
        if record :
            record.real_label = val
            ## save msg to database
            record.save()
        else:
            return JsonResponse({'code': 400, 'msg': 'Edit Result Has An Error ！！'})
    except :
        # pass
        return JsonResponse({'code': 400, 'msg': 'Edit Result Has An Error ！！'})
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Edit Result'})
