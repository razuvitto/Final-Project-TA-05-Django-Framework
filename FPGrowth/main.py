# Coded with <3 Razuvitto
# location : FPGrowth/main.py
# 2018-2019

from django.core.files.storage import default_storage
import pandas as pd
import numpy as np
import pyfpgrowth
import random
from operator import itemgetter
import time
import collections

def main(dataset_name):
    start = time.time()

        # Mining Closed Pattern Function
    def issubset(a,b):
        for x in a:
            if x not in b:
                return False
        return True

    dictionary = {}

    def close_pattern(freq_pattern):
        satu = []
        for a,b in freq_pattern.items():
            w = a
            flag = False
            for i,j in freq_pattern.items():
                if a!=i and i not in satu:
                    #print(a,b,i,j)
                    if issubset(a,i) and b == j:
                        flag = True
                        break
            #print(flag)
            if flag == False:
                dictionary[w] = b
            satu.append(w)

        return dictionary


    def CountItem(item):
        sums = 0
        for i in ListDataset:
            flag = True
            for x in item:
                if x not in i:
                    flag = False
                    break
            if flag == True:
                sums += 1
        return sums


    def CheckSupport(ListItems):
        support = []
        for i in ListItems:
            support.append(CountItem(i))
        return support


    def FindAntecedence(Rules):
        ante = []
        for key, value in Rules.items():
            TempKey = []
            TempValue = []
            SortedKey = sorted(key)
            SortedValue = sorted(value)

            for itemkey in SortedKey:
                TempKey.append(itemkey)

            for itemvalue in SortedValue:
                TempValue.append(itemvalue)

            ante.append(tuple(TempKey))

        return ante
        
    def FindConsequent(Rules):
        conse = []
        for key, value in Rules.items():
            TempKey = []
            TempValue = []
            SortedKey = sorted(key)
            SortedValue = sorted(value)

            for itemkey in SortedKey:
                TempKey.append(itemkey)

            for itemvalue in SortedValue:
                TempValue.append(itemvalue)

            conse.append(tuple(TempValue))

        return conse
        
    def FindAntecedenceConsequent(Rules):
        # // Create new dataframe for Association Rules //

        # Separate Association Rules Items ( 'Antecedence', 'Consequent', 'Confidence', 'Antecedence U Consequent')
        ante = []
        conse = []
        anteconse = []
        for key, value in Rules.items():
            TempKey = []
            TempValue = []
            SortedKey = sorted(key)
            SortedValue = sorted(value)
            for itemkey in SortedKey:
                TempKey.append(itemkey)
            for itemvalue in SortedValue:
                TempValue.append(itemvalue)

            ante.append(tuple(TempKey))
            conse.append(tuple(TempValue))

        for i in range(len(ante)):
            combine = (tuple(ante[i]+conse[i]))
            anteconse.append(combine)

        return anteconse

    def FindAnteConse(Rules):
        # // Create new dataframe for Association Rules //

        # Separate Association Rules Items ( 'Antecedence', 'Consequent', 'Confidence', 'Antecedence U Consequent')
        ante = []
        conse = []
            
        for key, value in Rules.items():
            ante.append(sorted(key))
            conse.append(sorted(value))
                
        result = []
        for i in range(len(ante)):
            Temp = 'If %s, Then %s' %(ante[i], conse[i])
            result.append(Temp)
                
        return result

    def Encode(list_of_rules):
        encode = []
        for i in list_of_rules:
            temp = []
            for j in semua_nilai_dari_atribut:
                if j in i[0]:
                    temp.append('00')
                elif j in i[1]:
                    temp.append('11')
                else:
                    temp.append('01')
            encode.append(temp)
        return encode

    def RouletteWheel(Individu):
        maxi = sum(Individu.values())
        pick = random.uniform(0, maxi)
        current = 0
        for key, value in Individu.items():
            current += value
            if current > pick:
                return key

    def CheckRules(list_encode ,role, index):
        return list_encode[index][role]

    def CountAnteConse(list_encode, role, index):
        value = list_encode[index].count(role)
        return value

    def ListOfMutationPoints(Offspring, MutationProbability):
        ListOfMutationPoint = []
        for i in range(len(Offspring)):
            RandomValue = random.sample(range(0, len(Offspring[0])), MutationProbability)
            ListOfMutationPoint.append(RandomValue)
        return ListOfMutationPoint

    def Decode(list_of_encode):
        decode = {}
        for i in range(len(list_of_encode)):
            ante_decode = []
            conse_decode = []
            for j in range(len(list_of_encode[i])):
                if list_of_encode[i][j] == '00':
                    ante_decode.append(semua_nilai_dari_atribut[j])
                elif list_of_encode[i][j] == '11':
                    conse_decode.append(semua_nilai_dari_atribut[j])
            result_ante = tuple(ante_decode)
            result_conse = tuple(conse_decode)
            decode[result_ante] = result_conse
        return decode





    # 1. Prepare Data ( Data Proprocessing )
        # Input dataset name
    input_dataset_name = dataset_name
           
    data_for_visualization = pd.read_csv('FPGrowth/dataset/'+input_dataset_name, low_memory=False)
    data_for_process = pd.read_csv('FPGrowth/dataset/'+input_dataset_name, low_memory=False)

    print("Step : Data Preprocessing (START)")
    # Check ID in Dataset
    
    ListOfId = []
    for i in data_for_visualization.columns:
        if (data_for_visualization[i].nunique()) == len(data_for_visualization):
            ListOfId.append(i)

    # Delete ID Column from Dataset
    for i in ListOfId:
        del data_for_process[i]
        del data_for_visualization[i]

        # 1.1. Data Cleaning
            # 1.1.1. Handle Missing Value
    data_for_process.fillna(method='ffill',inplace=True) # Fill Forward
    data_for_process.fillna(method='bfill',inplace=True) # Fill Backward
    data_for_visualization.fillna(method='ffill',inplace=True) # Fill Forward
    data_for_visualization.fillna(method='bfill',inplace=True) # Fill Backward

    # =========================== Data After Data Cleaning ===========================
    # data_for_visualization.to_csv('Data after Data Cleaning.csv', index = False)

    # Display Top 10 Row Preview Data After Data Cleaning
    preview_after_data_cleaning = data_for_visualization.head(10)

    nama_kolomnya = []
    for i in preview_after_data_cleaning.columns:
        nama_kolomnya.append(i)

    datanya = []
    for x in nama_kolomnya:
        temp = []
        for i in preview_after_data_cleaning[x]:
            data = i
            temp.append(data)
        datanya.append(temp)

    indexnya = []
    for i in range(len(nama_kolomnya)):
        indexnya.append(i)


    datanya2 = []

    for i in range(len(preview_after_data_cleaning)):
        Temp = []
        for ColumnName in preview_after_data_cleaning.columns:
            value = preview_after_data_cleaning[ColumnName][i]
            Temp.append(value)
        datanya2.append(Temp)


            # 1.1.2. Check for Numeric Attribute
    numerics = ['int_','int8','int16', 'int32', 'int64','uint8','uint16', 'uint32', 'uint64', 'float_', 'float16', 'float32', 'float64']
    attribute_numeric = data_for_visualization.select_dtypes(include=numerics)

            # 1.1.3. Check for Nominal Attribute
    attribute_nominal = data_for_visualization.select_dtypes(include = ['O'])

            # 1.1.4. Transform Nominal to Numeric
    for att_nom in attribute_nominal:
        data_for_visualization[att_nom]=data_for_visualization[att_nom].astype('category').cat.codes

    


            # 1.1.5. Check Correlation
    corr = data_for_visualization.corr()

            # if the correlation value = 1, remove it 
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False

    # select the attribute with correlation value less than 0,9
    selected_columns = data_for_visualization.columns[columns]

    # We get the attribute that we will use or process
    attribute_to_use = list(selected_columns)   

    # New Dataset with attribute based on attribute we get from visualization
    Dataset = data_for_process[attribute_to_use]
    Dataset_visual = data_for_visualization[attribute_to_use]
    Data_before_transformation = Dataset
    Data_visual_before_transformation = Dataset_visual

    # =========================== Data After Data Selection ===========================
    # Data_before_transformation.to_csv('Data after Data Selection.csv', index = False)

    # Display Top 10 Row Preview Data After Data Selection
    preview_after_data_selection = Data_before_transformation.head(10)
    
    nama_kolomnya_data_selection = []
    for i in preview_after_data_selection.columns:
        nama_kolomnya_data_selection.append(i)
    
    datanya_data_selection = []
    for x in nama_kolomnya_data_selection:
        temp = []
        for i in preview_after_data_selection[x]:
            data = i
            temp.append(data)
        datanya_data_selection.append(temp)

    DictOfPreviewDataAfterDataSelection = []
    for i in range(len(preview_after_data_selection)):
        TempDict = {}
        for ColumnName in preview_after_data_selection.columns:
            TempDict[ColumnName] = preview_after_data_selection[ColumnName][i]
        DictOfPreviewDataAfterDataSelection.append(TempDict)


        # 1.2. Binning
    # Check for attribute to binning which is attribute in new dataset with numeric values
    atribut_mau_dibinning = []
    for i in Dataset.columns:
        if i in attribute_numeric:
            atribut_mau_dibinning.append(i)


            # 1.2.1. Binning Process
    dataframelabel = []
    for att in atribut_mau_dibinning:
        atribut_name = Dataset[att]
        max_value = atribut_name.max()
        min_value = atribut_name.min()
        n = len(atribut_name)
        cbrt = n ** (1. / 3)
        data_describe = atribut_name.describe()
        Q1 = data_describe['25%']
        Q2 = data_describe['50%']
        Q3 = data_describe['75%']
        interquartile = Q3 - Q1
        
        # Bins
        bins = int(round((max_value - min_value) /  (2 *  (interquartile / cbrt))))

        # Binwidth
        binwidth = round((2 * (interquartile/cbrt)),2)
        
        # List of bins
        bins_list = []
        for bins_number in range(bins):
        #     bins_list.append("%.2f" % bins_number)
            x = bins_number * binwidth
            jumlah = round((float(max_value) - x),2)
            bins_list.append(jumlah)
        bins_list = sorted(bins_list, key=float)
        
        # List of bin label
        # bin labels
        bin_labels = []
        for label in range(int(bins)):
            value = att + ' ' +str(label)
            bin_labels.append(value)
            
        for i in range(1):
            df_label = pd.DataFrame({'Bins':bins_list,
                                'Label':bin_labels})
            
        dataframelabel.append(df_label)
        
        DataFrameBin = pd.DataFrame({'Bin': bins_list})
        
        # Binning Process
        # select value with header
        data_list = [('nums', Dataset[att])]

        # # Create the labels for the bin
        # bin_labels = ['Lat1','Lat2','Lat3','Lat4','Lat5','Lat6','Lat7','Lat8','Lat9']

        # Create the dataframe object using the data_list
        # df = pd.DataFrame.from_items(data_list)
        df = pd.DataFrame.from_dict(collections.OrderedDict(data_list))
        # Define the scope of the bins
        # ['bound1' , 'bound2' = 1]
        # ['bound2' , 'bound3' = 2]
        # ['bound3' , 'bound4' = 3]
        # ['bound4' , 'bound5' = 4]

        # bins = [bound1, bound2, bound3, bound4, bound5]

        # Create the "bins" column using the cut function
        df[att + '_bins'] = pd.cut(df['nums'], bins=bins, labels=bin_labels)
        
        del Dataset[att]
        Dataset[att + '_bins'] = df[att + '_bins']

            # 1.2.2. Label of Binning
    for f in range(len(atribut_mau_dibinning)):
        g = dataframelabel[f]
        # print(g)
        # g.to_csv('Nilai Binning %s.csv' %(f), index = False)

    LabelValue = []
    BinsValue = []
    for i in dataframelabel:
        for j in i.values:
            BinsValue.append(j[0])
            LabelValue.append(j[1])

    BinsDataFrame = pd.DataFrame({'Bins' : BinsValue,
                       'Label': LabelValue})


    ListOfDictOfBinning = []
    for i in range(len(BinsDataFrame)):
        x = {}
        x['Bins'] = BinsDataFrame['Bins'][i]
        x['Label'] = BinsDataFrame['Label'][i]
        ListOfDictOfBinning.append(x)


        # 1.3. Labelling
    # Check for attribute to labeled which is attribute in new dataset with nominal values
    atribut_mau_dilabel = []
    for att_mau_dilabel in Dataset.columns:
        if att_mau_dilabel in attribute_nominal:
            atribut_mau_dilabel.append(att_mau_dilabel)

            # 1.3.1. Labelling Process
    for atribut_label in atribut_mau_dilabel:
        Dataset[atribut_label] = atribut_label + ' ' + Dataset[atribut_label].astype(str)

    Dataset = Dataset

    print("Step : Data Preprocessing (DONE)")
    # =========================== Data After Data Transformation ===========================
    # Dataset.to_csv('Data after Data Preprocessing.csv', index = False)

    # Display Top 10 Row Preview Data After Data Transformation ( AFTER DATA PREPROCESSING )

    preview_after_data_transformation = Dataset.head(10)

    DictOfPreviewDataAfterDataTransformation = []
    for i in range(len(preview_after_data_transformation)):
        TempDict = {}
        for ColumnName in preview_after_data_transformation.columns:
            TempDict[ColumnName] = preview_after_data_transformation[ColumnName][i]
        DictOfPreviewDataAfterDataTransformation.append(TempDict)


    nama_kolomnya_data_transformation = []
    for i in preview_after_data_transformation.columns:
        nama_kolomnya_data_transformation.append(i)


    datanya_data_transformation = []
    for x in nama_kolomnya_data_transformation:
        temp = []
        for i in preview_after_data_transformation[x]:
            data = i
            temp.append(data)
        datanya_data_transformation.append(temp)

    print("Step : Frequent Pattern Mining (START)")
    # 2. Rules Mining [ FP-Growth Algorithm ]

        # 2.1. Frequent Pattern Mining
    # The data set to an array
    data = Dataset.values.tolist()
    # data = Dataset.T.apply(lambda x: x.dropna().tolist()).tolist()

    if len(Dataset) <= 200:
        minsup = 2
    else:
        minsup = round(0.01 * len(Dataset))
        
    FrequentPattern = pyfpgrowth.find_frequent_patterns(data, minsup)

    FrequentFrame = pd.DataFrame(list(FrequentPattern.items()), columns=['Itemsets', 'support'])
    print("Step : Frequent Pattern Mining (DONE)")
    # =========================== Frequent Pattern ===========================
    # FrequentFrame.to_csv('Frequent Pattern.csv' ,index=False)
    print("Step : Closed Pattern Mining (START)")

    # =========================== Closed Pattern ===========================
        # 2.2. Closed Pattern Mining
    ClosedPattern = close_pattern(FrequentPattern)
    ClosedFrame = pd.DataFrame(list(ClosedPattern.items()), columns=['Itemset', 'Support'])
    print("Step : Closed Pattern Mining (DONE)")
    
    # ClosedFrame.to_csv('Closed Pattern.csv', index=False)

        # 2.3. Association Rules Mining
    print("Step : Association Rules Mining (START)")
            # 2.3.1. Mining Association Rules
    AssociationRules = pyfpgrowth.generate_association_rules(ClosedPattern, 0.8)

            # 2.3.2. Separate Rules to Antecedence, Consequent and Antecedence Join Consequent
    # // Create new dataframe for Association Rules //
    # Separate Association Rules Items ( 'Antecedence', 'Consequent', 'Confidence', 'Antecedence U Consequent')
    ante = []
    conse = []
    anteconse = []
    confi = []
    for key, value in AssociationRules.items():
        TempKey = []
        TempValue = []
        SortedKey = sorted(key)
        SortedValue = sorted(value[0])
        for itemkey in SortedKey:
            TempKey.append(itemkey)
        for itemvalue in SortedValue:
            TempValue.append(itemvalue)
            
        ante.append(tuple(TempKey))
        conse.append(tuple(TempValue))
        confi.append(value[1])

    for i in range(len(ante)):
        combine = (tuple(ante[i]+conse[i]))
        anteconse.append(combine)

            # 2.3.3. Function to Count Support
    ListDataset = Dataset.values.tolist()

    separate_ante = []
    separate_conse = []
    for key, value in AssociationRules.items():
        separate_ante.append(key)
        separate_conse.append(value[0])
        
    RulesForGA = dict(zip(separate_ante, separate_conse))

    AssosiationFrame = pd.DataFrame({'Rules': FindAnteConse(RulesForGA),
                                     'Antecedence': FindAntecedence(RulesForGA) ,
                                     'Consequent': FindConsequent(RulesForGA) ,
                                     'sup(A)': CheckSupport(FindAntecedence(RulesForGA)),
                                     'sup(B)' : CheckSupport(FindConsequent(RulesForGA)),
                                     'sup(A U B)' : CheckSupport(FindAntecedenceConsequent(RulesForGA))
                                    })

    AssosiationFrame['Kulczynski'] = round((AssosiationFrame['sup(A U B)'] / 2) * ( ( 1/AssosiationFrame['sup(A)'] ) +  ( 1/AssosiationFrame['sup(B)'] )), 4)
    AssosiationFrame['IR'] = (abs(AssosiationFrame['sup(A)'] - AssosiationFrame['sup(B)'])) / (AssosiationFrame['sup(A)'] + AssosiationFrame['sup(B)'] - AssosiationFrame['sup(A U B)'])
    AssosiationFrame = AssosiationFrame.sort_values('Kulczynski', ascending=False)
    AssosiationFrame = AssosiationFrame[AssosiationFrame['sup(A)'] >= minsup]
    AssosiationFrame = AssosiationFrame[AssosiationFrame['sup(B)'] >= minsup]
    AssosiationFrame = AssosiationFrame[AssosiationFrame['sup(A U B)'] >= minsup]
    print("Step : Association Rules Mining (DONE)")

    ListOfDictOfRules = []
    const_len_rules = len(AssosiationFrame['Rules'])
    for i in range(const_len_rules):
        x = {}
        x['Rules'] = AssosiationFrame['Rules'][i]
        x['Antecedence'] = AssosiationFrame['Antecedence'][i]
        x['Consequent'] = AssosiationFrame['Consequent'][i]
        x['SupportAntecedence'] = AssosiationFrame['sup(A)'][i]
        x['SupportConsequent'] = AssosiationFrame['sup(B)'][i]
        x['SupportAntecedenceandConsequent'] = AssosiationFrame['sup(A U B)'][i]
        x['Kulczynski'] = AssosiationFrame['Kulczynski'][i]
        x['IR'] = AssosiationFrame['IR'][i]
        ListOfDictOfRules.append(x)



    # =========================== Beginning Rules ( From FP-Growth Algorithm ) ===========================
    # AssosiationFrame.to_csv('Hasil/Rules Awal.csv', index = False)

    semua_nilai_dari_atribut = []
    for nama_kolom in Dataset.columns:
        temp_data = sorted(Dataset[nama_kolom].unique())
        semua_nilai_dari_atribut.extend(temp_data)

    RulesFPGrowthForEncode = []
    for i,j in RulesForGA.items():
        valante = i
        valconse = j
        temp = valante, valconse
        RulesFPGrowthForEncode.append(temp)


    # ! INPUT FOR GA = RulesForGA

    AssosiationRulesDataFrameForGA = pd.DataFrame({'Rules': FindAnteConse(RulesForGA),
                                                   'Antecedence': FindAntecedence(RulesForGA) ,
                                                   'Consequent': FindConsequent(RulesForGA) ,
                                                   'sup(A)': CheckSupport(FindAntecedence(RulesForGA)),
                                                   'sup(B)' : CheckSupport(FindConsequent(RulesForGA)),
                                                   'sup(A U B)' : CheckSupport(FindAntecedenceConsequent(RulesForGA))
                                                  })

    AssosiationRulesDataFrameForGA['Kulczynski'] = round((AssosiationRulesDataFrameForGA['sup(A U B)'] / 2) * ( ( 1/AssosiationRulesDataFrameForGA['sup(A)'] ) +  ( 1/AssosiationRulesDataFrameForGA['sup(B)'] )), 4)
    AssosiationRulesDataFrameForGA['IR'] = (abs(AssosiationRulesDataFrameForGA['sup(A)'] - AssosiationRulesDataFrameForGA['sup(B)'])) / (AssosiationRulesDataFrameForGA['sup(A)'] + AssosiationRulesDataFrameForGA['sup(B)'] - AssosiationRulesDataFrameForGA['sup(A U B)'])
    AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA.sort_values('Kulczynski', ascending=False)
    AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(A)'] >= minsup]
    AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(B)'] >= minsup]
    AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(A U B)'] >= minsup]
    print("Step : Rules Optimization (START)")
    # Plot Kulc Score
    # KulcValue = AssosiationRulesDataFrameForGA['Kulczynski']

    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # Make a fake dataset:
    # height = KulcValue
    # bars = KulcValue.index.values
    # y_pos = np.arange(len(bars))
    
    # Create bars
    # plt.bar(y_pos, height)
    # Add title and axis names
    # plt.title('Kulczynski Values from Each Association Rules')
    # plt.xlabel('Association Rules Number')
    # plt.ylabel('Kulczynski Value')

    
    # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    # Show graphic
    # plt.show()

    CrossOverProbability = 0.8
    MutationProbability = 0.01
 
    BestAssociationRulesDataFrame = AssosiationRulesDataFrameForGA

    constanta = len(AssociationRules)

    
    # =========================================== !!! GENETIC ALGORITHM !!! ===========================================

    for step in range(1,501):    
        # Data Yang Mau Diolah Masuk
        BestAssociationRulesDataFrame
        AssosiationRulesDataFrameForGA = pd.DataFrame({'Rules': FindAnteConse(RulesForGA),
                                            'Antecedence': FindAntecedence(RulesForGA) ,
                                            'Consequent': FindConsequent(RulesForGA) ,
                                            'sup(A)': CheckSupport(FindAntecedence(RulesForGA)),
                                            'sup(B)' : CheckSupport(FindConsequent(RulesForGA)),
                                            'sup(A U B)' : CheckSupport(FindAntecedenceConsequent(RulesForGA))
                                            })

        AssosiationRulesDataFrameForGA['Kulczynski'] = round((AssosiationRulesDataFrameForGA['sup(A U B)'] / 2) * ( ( 1/AssosiationRulesDataFrameForGA['sup(A)'] ) +  ( 1/AssosiationRulesDataFrameForGA['sup(B)'] )), 4)
        AssosiationRulesDataFrameForGA['IR'] = (abs(AssosiationRulesDataFrameForGA['sup(A)'] - AssosiationRulesDataFrameForGA['sup(B)'])) / (AssosiationRulesDataFrameForGA['sup(A)'] + AssosiationRulesDataFrameForGA['sup(B)'] - AssosiationRulesDataFrameForGA['sup(A U B)'])
        AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA.sort_values('Kulczynski', ascending=False)
        AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(A)'] >= minsup]
        AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(B)'] >= minsup]
        AssosiationRulesDataFrameForGA = AssosiationRulesDataFrameForGA[AssosiationRulesDataFrameForGA['sup(A U B)'] >= minsup]
        
        # Hitung Fitness
        fitness = []
        for i in AssosiationRulesDataFrameForGA['Kulczynski']:
            value = float((("{:.6f}".format(i))))
            fitness.append(value)
        fitness = sorted(fitness, reverse=True)
        
        # Hitung total fitness
        total_fitness = round(sum(fitness),6)
        
        # Fitness Individu / Fitness Total
        fitness_per_total = []
        for i in fitness:
            temp = (round((i / total_fitness),6))
            fitness_per_total.append(temp)
        
        # Fitness Individu / Fitness Total displayed on DataFrame
        Fitness_DataFrame = pd.DataFrame({'Fitness Value': fitness,
                        'Total Fitness': total_fitness,
                        'Result': fitness_per_total})
        
        # Give ID for each fitness value
        IndexIndividual = []
        for i in range(len(fitness_per_total)):
            IndexIndividual.append(i)
        
        # ID and fitness to dict
        BeginningIndividual = dict(zip(IndexIndividual, fitness_per_total))
        
        # =============================================== SELECTION =============================================== 
        # Do Roulette Wheel
        RouletteResult = []
        for i in range(len(fitness)):
            temp = RouletteWheel(BeginningIndividual)
            if temp not in RouletteResult:
                RouletteResult.append(temp)
                
        IndividualValue = round(len(RouletteResult) * CrossOverProbability)

        # Update IndividualValue, If odd, IndividualValue - 1
        if IndividualValue % 2 == 0:
            IndividualValue = IndividualValue
        else:
            IndividualValue = IndividualValue - 1
        IndividualValue
        
        # Take some individu from RouletteResult based on IndividualValue
        RoulettePosition = RouletteResult[:IndividualValue]
        
        
        ListOfAntecedence = AssosiationRulesDataFrameForGA['Antecedence']
        ListOfConsequent  = AssosiationRulesDataFrameForGA['Consequent']
        
        ListAntecedence = []
        for i in ListOfAntecedence:
            ListAntecedence.append(i)

        ListConsequent = []
        for i in ListOfConsequent:
            ListConsequent.append(i)
            
        MyAssociationRules = dict(zip(ListAntecedence, ListConsequent))
        
        s = []
        for a,b in MyAssociationRules.items():
            d = []
            d.append(a)
            d.append(b)
            s.append(d)
            
        # Rules awal setelah seleksi adalah sebanyak RulesForCrossover, dan itu yang akan digunakan ke tahap berikutnya
        RulesForCrossover = []
        for i in RoulettePosition:
            RulesForCrossover.append(s[i])
            
        # Pada Tahap ini, Rules awal dianggap sebagai Rules yang Baik
        # Diberikan fitness baru pada masing2 Rules Baik untuk melihat siapa saja yang akan terkena crossover
        FitnessForCrossover = []
        for i in range(len(RulesForCrossover)):
            Temp = round(random.uniform(0,1), 1)
            FitnessForCrossover.append(Temp)
        
        # List Rules dengan ID, Nilai Fitness, dan Rulesnya
        RulesWithFitnessCrossover = []
        for i in range(len(RulesForCrossover)):
            Temp = i,FitnessForCrossover[i], RulesForCrossover[i]
            RulesWithFitnessCrossover.append(Temp)
        
        # Tidak semua rules hasil seleksi akan di crossover. Rules yang akan di crossover adalah rules yang fitnessnya lebih rendah dari CP
        RulesToCrossover = []
        for i in range(len(RulesWithFitnessCrossover)):
            if RulesWithFitnessCrossover[i][1] < CrossOverProbability:
                Temp = RulesWithFitnessCrossover[i][0], RulesWithFitnessCrossover[i][1], RulesWithFitnessCrossover[i][2]
                RulesToCrossover.append(Temp)

        RulesNotToCrossover = []
        for i in range(len(RulesWithFitnessCrossover)):
            if RulesWithFitnessCrossover[i][1] >= CrossOverProbability:
                Temp = RulesWithFitnessCrossover[i][2]
                RulesNotToCrossover.append(Temp)
        
    #     print ('Rules awal adalah',len(RulesWithFitnessCrossover))
    #     print ('dari',len(RulesWithFitnessCrossover),'Rules, ada',len(RulesToCrossover), 'yang mungkin di crossover.')
    #     print ('Kemudian, ada',len(RulesNotToCrossover), 'Rules yang tidak mungkin di crossover')
        if len(RulesToCrossover) > 0:
            
            RulesToCrossover = sorted(RulesToCrossover, key=itemgetter(1))
        
            RulesFixToCrossover = [ ]
            RulesCanceledToCrossover = [ ]
            
            
            if len(RulesToCrossover) % 2 == 1:
                RulesFixToCrossover = RulesToCrossover[:len(RulesToCrossover) - 1]
                RulesCanceledToCrossover.append(RulesToCrossover[-1])
            
                ListOfRulesCanceledToCrossover = []
                for i in RulesCanceledToCrossover:
                    Item = i[2]
                    ListOfRulesCanceledToCrossover.append(Item)
                RulesNotToCrossover = RulesNotToCrossover + ListOfRulesCanceledToCrossover
            else:
                RulesFixToCrossover = RulesToCrossover
    #             print('Rules yang fix mau di crossover adalah sebanyak' , len(RulesFixToCrossover))
                
                
            
            
        else:
    #         RulesNotToCrossover = RulesToCrossover
            RulesNotToCrossover = RulesNotToCrossover
            
    #     print ('Kemudian, dari',len(RulesToCrossover), 'rules yang mungkin di crossover, ada', len(RulesCanceledToCrossover),'Rules Batal Menjadi Calon Crossover')
            
        
    #     print ('Maka pada akhirnya akan ada',len(RulesFixToCrossover), 'Rules yang Akan di Crossover')
    #     print ('dan akan ada',len(RulesNotToCrossover), 'Rules yang Tidak di crossover')

        if len(RulesFixToCrossover) > 0:
        
            RulesForEncode = []
            for i in range(len(RulesFixToCrossover)):
                RulesForEncode.append(RulesFixToCrossover[i][2])
            RulesForEncode
        
            encode_result = Encode(RulesForEncode)
        
            # Pemasangan Individu
            Couple = []
            a = 0
            for i in range(int(len(RulesForEncode)/2)):
                b = a + 1
                c = [a,b]
                Couple.append(c)
                a = a + 1
                b = a + 1
                a = b
        
            RandomA = []
            RandomB = []
            for i in range(len(Couple)):
                RandomA.append(Couple[i][0])
                RandomB.append(Couple[i][1])
            
            ListEncodeByRandomA = [encode_result[i] for i in RandomA]
            ListEncodeByRandomB = [encode_result[i] for i in RandomB]
        
            CoupleForCrossover = tuple(zip(ListEncodeByRandomA, ListEncodeByRandomB))

            # Check Couple to Crossover
            # Pasangan dapat dipastikan dengan potongan kode dibawah
        #     for i in range(len(CoupleForCrossover)):
        #         print(Decode(CoupleForCrossover[i]))

            # Make a cutpoint for crossover
            TwoPoint = []
            for i in range(len(CoupleForCrossover)):
                RandomPoint = random.sample(range(1, len(CoupleForCrossover[0][0])), 2)
                TwoPoint.append(sorted(RandomPoint))
            
            Offspring = []
            const = len(CoupleForCrossover[0][0])
            for i in range(len(CoupleForCrossover)):
                left,right = [],[]
                for x in range(0, TwoPoint[i][0]):
                    left.append(CoupleForCrossover[i][0][x])
                    right.append(CoupleForCrossover[i][1][x])
                for q in range(TwoPoint[i][0], TwoPoint[i][1]):
                    left.append(CoupleForCrossover[i][1][q])
                    right.append(CoupleForCrossover[i][0][q])
                for k in range(TwoPoint[i][1], const):
                    left.append(CoupleForCrossover[i][0][k])
                    right.append(CoupleForCrossover[i][1][k])
                Offspring.append(left)
                Offspring.append(right)
            
            # Make sure Offspring from Crossover is not fail
            NewOffspringAfterCrossover = []
            for i in range(len(Offspring)):
                TempOffspring = Offspring[i]
                if ('00') in TempOffspring and ('11') in TempOffspring:
                    NewOffspringAfterCrossover.append(TempOffspring)
            NewOffspringAfterCrossover
        
            BadRulesAfterCrossover = Decode(NewOffspringAfterCrossover)
        else:
            TempRules = Encode(RulesForCrossover)
            OriginalRules = Decode(TempRules)
            BadRulesAfterCrossover = OriginalRules
            
            
        AnteListOfBadRulesAfterCrossover = []
        ConseListOfBadRulesAfterCrossover = []
        for key,value in BadRulesAfterCrossover.items():
            AnteListOfBadRulesAfterCrossover.append(key)
            ConseListOfBadRulesAfterCrossover.append(value)
            
        ListOfBadRulesAfterCrossover = []
        for i in range(len(BadRulesAfterCrossover)):
            Rules = [AnteListOfBadRulesAfterCrossover[i], ConseListOfBadRulesAfterCrossover[i]]
            ListOfBadRulesAfterCrossover.append(Rules)
        
        # Join Offspring From Crossover to RulesNotToCrossover
        OffspringFromCrossover = ListOfBadRulesAfterCrossover + RulesNotToCrossover

        EncodeForMutation = Encode(OffspringFromCrossover)

        FitnessForMutation = []
        for i in EncodeForMutation:
            Temp = []
            for j in range(len(i)):
                Temp.append(round(random.uniform(0,1), 3))
            FitnessForMutation.append(Temp)
            
        RulesToMutation = []
        RulesNotToMutation = []
        for i in range(len(FitnessForMutation)):
            if min(FitnessForMutation[i]) < MutationProbability:
                RulesToMutation.append(EncodeForMutation[i])
            else:
                RulesNotToMutation.append(EncodeForMutation[i]) 
        
        if len(RulesToMutation) > 0:
            ProbabilityMutation = round((len(RulesToMutation) * len(RulesToMutation[0]) * MutationProbability) / len(RulesToMutation))
            # print('Probability Mutation without divide', round((len(RulesToMutation) * len(RulesToMutation[0]) * MutationProbability)))
            # print('Probability Mutation', ProbabilityMutation)
            # print('Probability Mutation without Round', (len(RulesToMutation) * len(RulesToMutation[0]) * MutationProbability))
            # print('Rules To Mutation', len(RulesToMutation))
            # print('Rules Not To Mutation', len(RulesNotToMutation))
    #         print('Rules to Mutation',RulesToMutation)
    #         print('=============================================================================================================')
            MutationPoint = ListOfMutationPoints(RulesToMutation, ProbabilityMutation)
            # print('Mutation Point', MutationPoint)
    #         print('Mutation Point', MutationPoint)
    #         print('=============================================================================================================')
            # MUTASI
            change = {
                        '00' : '11',
                        '01' : '10',
                        '10' : '01',
                        '11' : '00'
                    }

            for i in range(len(MutationPoint)):
                for j in range(len(MutationPoint[i])):
                    RulesToMutation[i][MutationPoint[i][j]] = change[RulesToMutation[i][MutationPoint[i][j]]]



            NewOffspringAfterMutation = []
            for i in range(len(RulesToMutation)):
                TempOffspring = RulesToMutation[i]
                if ('00') in TempOffspring and ('11') in TempOffspring:
                    NewOffspringAfterMutation.append(TempOffspring)

    #         print('Offspring From Mutation', NewOffspringAfterMutation)
    #         print('=============================================================================================================')
            OffspringFromMutation = RulesNotToMutation + NewOffspringAfterMutation
        else:
            OffspringFromMutation = OffspringFromCrossover    


        RulesAfterGA = Decode(OffspringFromMutation)
        
        AssosiationRulesDataFrameAfterGA = pd.DataFrame({'Rules': FindAnteConse(RulesAfterGA),
                                            'Antecedence': FindAntecedence(RulesAfterGA),
                                            'Consequent': FindConsequent(RulesAfterGA),
                                            'sup(A)': CheckSupport(FindAntecedence(RulesAfterGA)),
                                            'sup(B)' : CheckSupport(FindConsequent(RulesAfterGA)),
                                            'sup(A U B)' : CheckSupport(FindAntecedenceConsequent(RulesAfterGA))
                                            })
        
        AssosiationRulesDataFrameAfterGA['Kulczynski'] = round((AssosiationRulesDataFrameAfterGA['sup(A U B)'] / 2) * ( ( 1/AssosiationRulesDataFrameAfterGA['sup(A)'] ) +  ( 1/AssosiationRulesDataFrameAfterGA['sup(B)'] )), 4)
        AssosiationRulesDataFrameAfterGA['IR'] = (abs(AssosiationRulesDataFrameAfterGA['sup(A)'] - AssosiationRulesDataFrameAfterGA['sup(B)'])) / (AssosiationRulesDataFrameAfterGA['sup(A)'] + AssosiationRulesDataFrameAfterGA['sup(B)'] - AssosiationRulesDataFrameAfterGA['sup(A U B)'])
        AssosiationRulesDataFrameAfterGA = AssosiationRulesDataFrameAfterGA.sort_values('Kulczynski', ascending=False)
        AssosiationRulesDataFrameAfterGA = AssosiationRulesDataFrameAfterGA[AssosiationRulesDataFrameAfterGA['sup(A)'] >= minsup]
        AssosiationRulesDataFrameAfterGA = AssosiationRulesDataFrameAfterGA[AssosiationRulesDataFrameAfterGA['sup(B)'] >= minsup]
        AssosiationRulesDataFrameAfterGA = AssosiationRulesDataFrameAfterGA[AssosiationRulesDataFrameAfterGA['sup(A U B)'] >= minsup]
        
    #     AssosiationRulesDataFrameAfterGA.to_csv("Hasil/Hasil Optimasi Iterasi ke - " + str(step) + ".csv", index = False)


    #     DataFrame untuk Offspring namanya AssosiationRulesDataFrameAfterGA         
        new_df = pd.concat([AssosiationRulesDataFrameAfterGA, BestAssociationRulesDataFrame])
        new_df = new_df.drop_duplicates()
        new_df_sorted = new_df.sort_values('Kulczynski', ascending=False)
        BestRules = new_df_sorted.head(constanta)
        # Kondisi Rules
        # BestRules.to_csv("Hasil/Final Rules/Kondisi Rules Iterasi ke - " + str(step) + ".csv", index = False)
        NewAntecedenceForIteration = BestRules['Antecedence']
        NewConsequentForIteration = BestRules['Consequent']
        DictOfNewRules = dict(zip(NewAntecedenceForIteration, NewConsequentForIteration))
        RulesForGA = DictOfNewRules
        BestAssociationRulesDataFrame = BestRules


    print("Step : Rules Optimization (DONE)")

    BestRules = BestRules.reset_index(drop=True)

    ListOfDictOfRulesAfterOptimization = []
    length_of_new_rules_after_optimization = len(BestRules['Rules'])
    for i in range(length_of_new_rules_after_optimization):
        x = {}
        x['Rules'] = BestRules['Rules'][i]
        x['Antecedence'] = BestRules['Antecedence'][i]
        x['Consequent'] = BestRules['Consequent'][i]
        x['SupportAntecedence'] = BestRules['sup(A)'][i]
        x['SupportConsequent'] = BestRules['sup(B)'][i]
        x['SupportAntecedenceandConsequent'] = BestRules['sup(A U B)'][i]
        x['Kulczynski'] = BestRules['Kulczynski'][i]
        x['IR'] = BestRules['IR'][i]
        ListOfDictOfRulesAfterOptimization.append(x)


    # ======================================================== VARIABLE TO RETURN ========================================================
    PlotOfKulcBeforeOptimize = list(AssosiationFrame['Kulczynski'])
    PlotOfKulcAfterOptimize = list(BestRules['Kulczynski'])
    PointPlotFP = []
    PointPlotGA = []
    for i in range(len(PlotOfKulcAfterOptimize)):
        val = [i, PlotOfKulcAfterOptimize[i]]
        val2 = [i, PlotOfKulcBeforeOptimize[i]]
        PointPlotFP.append(val)
        PointPlotGA.append(val2)

    HTMLFrequentPattern = FrequentPattern.items()
    HTMLClosedPattern = ClosedPattern.items()
    LengthOfFrequentPattern = len(FrequentFrame)
    LengthOfClosedPattern = len(ClosedPattern)
    LengthOfAssociationRules = len(AssociationRules)
    LengthOfAssociationRulesAfterOptimization = len(BestRules)
    LengthOfDataset = len(Dataset)
    DecreasePresentation = round(((LengthOfFrequentPattern  - LengthOfClosedPattern) / LengthOfFrequentPattern ) * 100 , 2)
    end = time.time()
    TimeExecution = end - start
    TimeExecution = round(TimeExecution, 9)

    return LengthOfDataset, LengthOfFrequentPattern, LengthOfClosedPattern, LengthOfAssociationRules, LengthOfAssociationRulesAfterOptimization, TimeExecution, DecreasePresentation, HTMLFrequentPattern, HTMLClosedPattern, ListOfDictOfRules, ListOfDictOfRulesAfterOptimization, PlotOfKulcBeforeOptimize, PlotOfKulcAfterOptimize, PointPlotFP, PointPlotGA, ListOfDictOfBinning, dataset_name, nama_kolomnya, datanya, indexnya, nama_kolomnya_data_selection, datanya_data_selection, nama_kolomnya_data_transformation, datanya_data_transformation  

# End of File