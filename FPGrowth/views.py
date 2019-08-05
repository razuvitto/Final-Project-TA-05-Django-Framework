from django.shortcuts import render
from FPGrowth import main

def index(request):
    if request.method == 'POST':
        global content
        text = request.POST['dataset_name']
        LengthOfDataset, LengthOfFrequentPattern, LengthOfClosedPattern, LengthOfAssociationRules, LengthOfAssociationRulesAfterOptimization, TimeExecution, DecreasePresentation, HTMLFrequentPattern, HTMLClosedPattern, ListOfDictOfRules, ListOfDictOfRulesAfterOptimization, PlotOfKulcBeforeOptimize, PlotOfKulcAfterOptimize, PointPlotFP, PointPlotGA, ListOfDictOfBinning, dataset_name, nama_kolomnya, datanya, indexnya, nama_kolomnya_data_selection, datanya_data_selection, nama_kolomnya_data_transformation, datanya_data_transformation   = main.main(text)
        content = {'LengthOfDataset': LengthOfDataset, 'LengthOfFrequentPattern': LengthOfFrequentPattern, 'LengthOfClosedPattern': LengthOfClosedPattern, 'LengthOfAssociationRules': LengthOfAssociationRules, 'LengthOfAssociationRulesAfterOptimization': LengthOfAssociationRulesAfterOptimization, 'TimeExecution': TimeExecution, 'DecreasePresentation': DecreasePresentation, 'HTMLFrequentPattern': HTMLFrequentPattern, 'HTMLClosedPattern': HTMLClosedPattern, 'ListOfDictOfRules': ListOfDictOfRules, 'ListOfDictOfRulesAfterOptimization': ListOfDictOfRulesAfterOptimization, 'PlotOfKulcBeforeOptimize': PlotOfKulcBeforeOptimize, 'PlotOfKulcAfterOptimize': PlotOfKulcAfterOptimize, 'PointPlotFP': PointPlotFP, 'PointPlotGA': PointPlotGA, 'ListOfDictOfBinning': ListOfDictOfBinning, 'dataset_name': dataset_name, 'nama_kolomnya': nama_kolomnya, 'datanya': datanya, 'indexnya': indexnya, 'nama_kolomnya_data_selection': nama_kolomnya_data_selection, 'datanya_data_selection':datanya_data_selection, 'nama_kolomnya_data_transformation': nama_kolomnya_data_transformation, 'datanya_data_transformation': datanya_data_transformation }
      #  request.session['content'] = content
        return render(request, 'mainpage.html', content)
    return render(request, 'index.html')
    
def frequent(request):
    return render(request,'frequent.html', content)
    
def closed(request):
    return render(request,'closed.html', content)

def rules(request):
    return render(request,'rules.html', content)

def optimizerules(request):
    return render(request,'rulesGA.html', content)

def mainpage(request):
    return render(request,'mainpage.html', content)

def compareresult(request):
    return render(request,'compare.html', content)

def documentationbinning(request):
    return render(request,'documentationdata.html', content)

def datacleaning(request):
    return render(request,'previewdatacleaning.html', content)

def dataselection(request):
    return render(request,'previewdataselection.html', content)

def datatransformation(request):
    return render(request,'previewdatatransformation.html', content)





