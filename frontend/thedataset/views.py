# return data
from django.shortcuts import render
# download file
from django.http import FileResponse
# return html
from django.views.decorators.clickjacking import xframe_options_exempt
# return Json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# save and read file
import os
# judge platform
import platform
# time
from django.utils import timezone
# delete file
import shutil , stat
# database tables
from . import models
# page helper
from django.core.paginator import Paginator

# Create your views here.
@xframe_options_exempt
def toDatasetList(request):
    return render(request,'thedataset/thedataset-list.html')


@csrf_exempt
def getDatasetList(request):
    type_select = request.GET.get('type_select')
    dataset_name_select = request.GET.get("dataset_name_select")
    domain_select = request.GET.get("domain_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")

    if dataset_name_select == None:
        dataset_name_select = ''
    if domain_select == None:
        domain_select = ''
    
    datasetList = models.DataSet.objects.values().filter(dataset_name__contains=dataset_name_select,domain__contains=domain_select).order_by('dataset_id')
    # print('type_select======',type_select)
    if type_select != '5':
        datasetList = datasetList.filter(type=int(type_select))    
    
    count = datasetList.count()

    # 分页
    paginator = Paginator(datasetList, limit)
    datasetList = paginator.get_page(page)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(datasetList)
    
    return JsonResponse(result)


@xframe_options_exempt
def toAddHtml(request):
    return render(request,'thedataset/thedataset-add.html')

@csrf_exempt
def addDataset(request):
    ##get base msg
    dataset_name = request.POST['dataset_name']
    domain = request.POST['domain']
    class_num = request.POST['class_num']
    source = request.POST['source']
    sample_total_num = request.POST['sample_total_num']
    sample_training_num = request.POST['sample_training_num']
    sample_validation_num = request.POST['sample_validation_num']
    sample_test_num = request.POST['sample_test_num']
    sentence_max_length = request.POST['sentence_max_length']
    sentence_avg_length = request.POST['sentence_avg_length']
    type = 1    # upload type == 1(User)
    ## get files
    file_all = request.FILES.get('file_all')

    if file_all == None:
        return JsonResponse(
            {'code': 201, 'msg': 'Please Chose file:all.tsv !!!'})
  
    if platform.system() == 'Linux':
        local_path = os.getcwd()+'/data/dataset/' + dataset_name +'/'
    elif platform.system() == 'Windows':
        local_path = os.getcwd()+"\data\dataset\\" + dataset_name +'\\'
    ## determine if the dataset exists
    if os.path.exists(local_path):
        return JsonResponse({'code':201,'msg':'The Dataset "'+ dataset_name +'" Already Exists , Please Check It !!!'})
    try:
        print(local_path)
        ## create local_path
        os.makedirs(local_path)

        ## save files
        destination = open(local_path + 'all.tsv', 'wb+')
        for chunk in file_all.chunks():
            destination.write(chunk)
        destination.close()



        # ## save msg to Database
        insert = models.DataSet.objects.create(dataset_name=dataset_name, domain=domain,
                                               class_num=class_num, source=source,
                                               local_path=local_path, type=type,
                                               sample_total_num=sample_total_num,
                                               sample_training_num=sample_training_num,
                                               sample_validation_num=sample_validation_num,
                                               sample_test_num=sample_test_num,
                                               sentence_avg_length=sentence_avg_length,
                                               sentence_max_length=sentence_max_length,
                                               create_time=timezone.now().strftime("%Y-%m-%d %H:%M:%S"))
        insert.save()
    except :
        if os.path.exists(local_path):
            os.removedirs(local_path)
        # pass
        return JsonResponse({'code': 400, 'msg': 'Add Dataset Has An Error ！！'})

    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Add Dataset'})

@xframe_options_exempt
def details(request):
    dataset_id = request.GET.get('dataset_id')
    print(dataset_id)
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    return render(request,'thedataset/thedataset-details.html',{'obj':obj})

def downloadDataset(request):
    name = request.GET.get('name')
    file_name = request.GET.get('file')
    if platform.system() == 'Linux':
        file = open(os.getcwd()+'/data/dataset/' + name + '/' + file_name, 'rb')
    elif platform.system() == 'Windows':
        file = open(os.getcwd()+'\data\dataset\\' + name + '\\' + file_name, 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename='+file_name
    return response

@xframe_options_exempt
def toEdit(request):
    dataset_id = request.GET.get('dataset_id')
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    return render(request,'thedataset/thedataset-edit.html',{'obj':obj})

@csrf_exempt
def editData(request):
    ##get base msg
    dataset_id = request.POST['dataset_id']
    dataset_name = request.POST['dataset_name']
    domain = request.POST['domain']
    class_num = request.POST['class_num']
    source = request.POST['source']
    sample_total_num = request.POST['sample_total_num']
    sample_training_num = request.POST['sample_training_num']
    sample_validation_num = request.POST['sample_validation_num']
    sample_test_num = request.POST['sample_test_num']
    sentence_max_length = request.POST['sentence_max_length']
    sentence_avg_length = request.POST['sentence_avg_length']

    try:
        # ## save msg to Database
        record = models.DataSet.objects.get(dataset_id=dataset_id)
        if record and record.dataset_name == dataset_name:
            record.domain = domain
            record.class_num = class_num
            record.source = source
            record.sample_total_num = sample_total_num
            record.sample_training_num = sample_training_num
            record.sample_validation_num = sample_validation_num
            record.sample_test_num = sample_test_num
            record.sentence_max_length = sentence_max_length
            record.sentence_avg_length = sentence_avg_length
            ## save msg to database
            record.save()
        else:
            return JsonResponse({'code': 400, 'msg': 'Edit Dataset Has An Error ！！'})
    except :
        # pass
        return JsonResponse({'code': 400, 'msg': 'Edit Dataset Has An Error ！！'})
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Edit Dataset'})

@csrf_exempt
def delData(request):
    ##get base msg
    print('there is deldate1')
    dataset_name = request.POST['dataset_name']
    
    print(dataset_name)
    print('there is deldate2')
    ## get file path
    if platform.system() == 'Linux':
        local_path = os.getcwd() + '/data/dataset/' + dataset_name + '/'
    elif platform.system() == 'Windows':
        local_path = os.getcwd() + '\data\dataset\\' + dataset_name + '\\'
    try:
        print(local_path)
        
        ## delete data in database
        models.DataSet.objects.filter(dataset_name=dataset_name).delete()
        ## delete data in disk
        for fileList in os.walk(local_path):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
            os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(local_path)
        # os.removedirs(local_path)
        
        
    except:
        # pass
        # return HttpResponseRedirect('/dataset_management/list')
        return JsonResponse({'code': 400, 'msg': 'Delete Dataset Has An Error ！！'})
   
    # return HttpResponseRedirect('/dataset_management/list')
    return JsonResponse({'code':200,'msg':'Successfully Delete Dataset'})









