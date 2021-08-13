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
import os, sys
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


base_dataset_path_linux = '/../textoir/data/'
base_dataset_path_win = '\\..\\textoir\\data\\'

# Create your views here.
@xframe_options_exempt
def toDatasetList(request):
    return render(request,'thedataset/thedataset-list.html')

def update_source(request):
    result = {}
#https://github.com/PolyAI-LDN/task-specific-datasets

    hyper_parameters = {

            'num_train_epochs': 5000,
            'feat_dim': 2000,
            'batch_size': 4096,

        }
    des={   
            'num_train_epochs_DEC': 'The number of epochs for training DEC model.',
            'num_train_epochs_SAE': 'The number of epochs for training stacked auto-encoder.',
            'num_train_epochs_DCN': 'The number of epochs for training DCN model.',
            'feat_dim' : 'The feature dimension.',
            'update_interval' : 'The number of int,ervals between contiguous updates.',
            'lr' : 'The learning rate for training DCN.',
            'momentum' : 'The momentum value of SGD optimizer.',
            'tol' : 'The tolerance threshold to stop training for DCN.',
            'batch_size':'Batch size',




            'max_num_words' : 'The maximum number of words.',
            'update_interval' : 'The number of intervals between contiguous updates.',
            'alpha' : 'The weights for updating the auxiliary distribution targets.',
            'num_warmup_train_epochs': 'The number of warmup training epochs.',
            'num_pretrain_epochs':'The number of pre-training epochs.',
            'lr_pre' : 'The learning rate for pre-training the backbone.',
            'num_refine_epochs': 'The number of refining epochs.',
            'u' : 'The upper bound of the dynamic threshold.',
            'l' : 'The lower bound of the dynamic threshold.',

            'weibull_tail_size': 'The factor of weibull model.',
            'alpharank' : 'The factor of alpha rank.',
        
            'threshold' :'The probability threshold for detecting the open samples.',
            'scale' : 'The scale factor of DOC.',
            'num_train_epochs': 'The number of training epochs.',
            'feat_dim' : 'The feature dimension.',
            'warmup_proportion' : 'The warmup ratio for learning rate.',
            'n_neighbors' : 'The number of neighbors of LOF.',
            'contamination' : 'The contamination factor of LOF.',
            'lr' : 'The learning rate of backbone.',
            'train_batch_size' : 'The batch size for training.',
            'eval_batch_size' : 'The batch size for evaluation. ',
            'test_batch_size' : 'The batch size for testing.',
            'wait_patient' : 'Patient steps for Early Stop.'
    }
    '''
    for k,v in hyper_parameters.items():
        

        add_prarm = models.Hyper_parameters.objects.create(
                                                param_name=k, 
                                                param_describe=des[k],  
                                                default_value=v,  
                                                value_type='float',
                                                model_id=15
    
                                                )
    '''
    #add_prarm.save()   #添加参数
    
     
    b=[]
    #for i in [149,150,151,152,153,154,155,156]:
        #a = models.Hyper_parameters.objects.filter(param_id=i).delete()#.first()#.all()
        #a.model_id=3
        #a.save()
    
    #update = models.Model_Tdes.objects.filter(model_name="OpenMax").first()#.all()
    #update.paper_source=''
    
    #update.save()
    #for dataset in datasetList:
        #dataset_list.append(dataset['model_id'])
        #dataset_list.append(dataset['model_name'])
    result['code'] = 0
    #result['msg'] =list( update)

#model_id  model_name    1, "ADB", 2, "DeepUnk", 3, "DOC", 4, "MSP", 5, "OpenMax", 
# 6, "CDACPlus", 7, "DeepAligned", 8, "DTC_BERT", 9, "KCL_BERT", 10, "MCL_BERT", 
# 11, #"AG", 12, "DCN", 13, "DEC", 14, "KM", 15, "SAE"
    
    return JsonResponse(list(models.Model_Tdes.objects.values().all()))
@csrf_exempt
def getDatasetList(request):
    type_select = '5'
    dataset_name_select = request.GET.get("dataset_name_select")
    domain_select = request.GET.get("domain_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    dataset_id={1,4,5,}

    if dataset_name_select == None:
        dataset_name_select = ''
    if domain_select == None:
        domain_select = ''
    
    datasetList = models.DataSet.objects.values().filter(dataset_name__contains=dataset_name_select,domain__contains=domain_select).order_by('dataset_id').exclude(dataset_id=7 ).exclude(dataset_id=6 )
    # print('type_select======',type_select)
    for i in datasetList:
        print(i)
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
    # file_all = request.FILES.get('file_all')
    file_name_list = ['file_train', 'file_dev', 'file_test']
    file_list_dic = {}
    for file_name in file_name_list:
        file_list_dic[file_name] = request.FILES.get(file_name)
        if file_list_dic[file_name] == None:
            return JsonResponse({'code': 201, 'msg': 'Please Chose file:'+ file_name.split('_')[1] +'.tsv !!!'})
    # file_train = request.FILES.get('file_train')
    # file_dev = request.FILES.get('file_dev')
    # file_test = request.FILES.get('file_test')


    # if file_all == None:
    #     return JsonResponse(
    #         {'code': 201, 'msg': 'Please Chose file:all.tsv !!!'})

    # if file_train == None:
    #     return JsonResponse(
    #         {'code': 201, 'msg': 'Please Chose file:train.tsv !!!'})
    # if file_dev == None:
    #     return JsonResponse(
    #         {'code': 201, 'msg': 'Please Chose file:vali.tsv !!!'})
    # if file_test == None:
    #     return JsonResponse(
    #         {'code': 201, 'msg': 'Please Chose file:test.tsv !!!'})
    if platform.system() == 'Linux':
        local_path = sys.path[0]+ base_dataset_path_linux + dataset_name +'/'
    elif platform.system() == 'Windows':
        local_path = sys.path[0] + base_dataset_path_win + dataset_name +'\\'
    ## determine if the dataset exists
    if os.path.exists(local_path):
        return JsonResponse({'code':201,'msg':'The Dataset "'+ dataset_name +'" Already Exists , Please Check It !!!'})
    try:
        print(local_path)
        ## create local_path
        os.makedirs(local_path)

        ## save files
        for file_name in file_name_list:
            destination = open(local_path + file_name.split('_')[1] +'.tsv', 'wb+')
            for chunk in file_list_dic[file_name].chunks():
                destination.write(chunk)
            destination.close()
            
        # destination = open(local_path + 'all.tsv', 'wb+')
        # for chunk in file_all.chunks():
        #     destination.write(chunk)
        # destination.close()
        # destination = open(local_path + 'train.tsv', 'wb+')
        # for chunk in file_train.chunks():
        #     destination.write(chunk)
        # destination.close()
        # destination = open(local_path + 'dev.tsv', 'wb+')
        # for chunk in file_dev.chunks():
        #     destination.write(chunk)
        # destination.close()
        # destination = open(local_path + 'test.tsv', 'wb+')
        # for chunk in file_test.chunks():
        #     destination.write(chunk)
        # destination.close()

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
            for i_path in os.listdir(local_path):
                os.removedirs(i_path)
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
            return JsonResponse({'code': 400, 'msg': 'Edit Dataset Has An Error1 ！！'})
    except :
        # pass
        return JsonResponse({'code': 400, 'msg': 'Edit Dataset Has An Error2 ！！'})
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Edit Dataset'})

@csrf_exempt
def delData(request):
    ##get base msg
    print('there is deldate1')
    dataset_name = request.POST['dataset_name']
    dataset_id = request.POST['dataset_id']
    

    record = models.DataSet.objects.get(dataset_id=dataset_id)
    if record.type == 0 :
        return JsonResponse({'code': 400, 'msg': 'Cannot Delete Internal Dataset ！！'})
    
    print(dataset_name)
    print('there is deldate2')
    ## get file path
    
    if platform.system() == 'Linux':
        local_path = sys.path[0]+ base_dataset_path_linux + dataset_name +'/'
    elif platform.system() == 'Windows':
        local_path = sys.path[0] + base_dataset_path_win + dataset_name +'\\'
    print('*'*20, '\n'*4, 'Delete Begin\t\t', local_path)
    try:
        
        ## delete data in disk
        if os.path.exists(local_path):
            print('*'*20, '\n'*4, 'Delete Begin\t\t', local_path)
            for i_path in os.listdir(local_path):
                print('*'*20, '\n'*4, 'Delete runing\t\t', os.path.join(local_path, i_path))
                os.remove(os.path.join(local_path, i_path))
                # os.removedirs(os.path.join(local_path, i_path))
            os.removedirs(local_path)
        # for fileList in os.walk(local_path):
        #     for name in fileList[2]:
        #         os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
        #     os.remove(os.path.join(fileList[0], name))
        # shutil.rmtree(local_path)
        # os.removedirs(local_path)
        
        ## delete data in database
        models.DataSet.objects.filter(dataset_id=dataset_id).delete()
    except:
        # pass
        # return HttpResponseRedirect('/dataset_management/list')
        return JsonResponse({'code': 400, 'msg': 'Delete Dataset Has An Error ！！'})
   
    # return HttpResponseRedirect('/dataset_management/list')
    return JsonResponse({'code':200,'msg':'Successfully Delete Dataset'})









