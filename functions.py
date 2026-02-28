#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rdkit import Chem
import pandas as pd


# In[ ]:


def makeQuery_DBselect(properties=[],iddp=None,DOI=None,with_DOI=False):
    if with_DOI:
        SQL = """
                SELECT * FROM (
                    SELECT 
                        `R`.`DOI` AS `DOI`,
                        `R`.`title` AS `title`,
                        `D`.`Chromophore_name` AS `Chromophore_name`,
                        `C`.`Smiles` AS `Chromophore_smiles`,
                        `D`.`Solvent_name` AS `Solvent_name`,
                        `S`.`Smiles` AS `Solvent_smiles`{0}
                    FROM
                        {1}((((SELECT * FROM `deep4chem`.`Datapoints` {4}) `D`
                        {2}
                        LEFT JOIN `deep4chem`.`Smiles` `C` ON ((`D`.`idChromophores` = `C`.`idSmiles`)))
                        LEFT JOIN `deep4chem`.`Smiles` `S` ON ((`D`.`idSolvents` = `S`.`idSmiles`)))
                        LEFT JOIN `deep4chem`.`Refers` `R` ON ((`D`.`idRefers` = `R`.`idRefers`)))
                    ) `FIN` WHERE{3}"""
    else:
        SQL = """
                SELECT * FROM (
                    SELECT 
                        `D`.`Chromophore_name` AS `Chromophore_name`,
                        `C`.`Smiles` AS `Chromophore_smiles`,
                        `D`.`Solvent_name` AS `Solvent_name`,
                        `S`.`Smiles` AS `Solvent_smiles`{0}
                    FROM
                        {1}(((SELECT * FROM `deep4chem`.`Datapoints` {4}) `D`
                        {2}
                        LEFT JOIN `deep4chem`.`Smiles` `C` ON ((`D`.`idChromophores` = `C`.`idSmiles`)))
                        LEFT JOIN `deep4chem`.`Smiles` `S` ON ((`D`.`idSolvents` = `S`.`idSmiles`)))
                    ) `FIN` WHERE{3}"""

    select_column = ""
    brackets = ""
    joins = ""
    wheres = ""
    for i,p in enumerate(properties):
        select_column+=",\n"
        select_column+=f"""\t`p{i}`.`Value` As `{p}` """
        
        brackets+="("

        joins+=f"""
LEFT JOIN (SELECT 
    `deep4chem`.`Property_values`.`idDatapoints` AS `idDatapoints`,
    `deep4chem`.`Property_values`.`Value` AS `Value`
FROM
    `deep4chem`.`Property_values`
WHERE
    (`deep4chem`.`Property_values`.`Property` = '{p}')) `p{i}` ON ((`D`.`idDatapoints` = `p{i}`.`idDatapoints`)))
"""
        if i>0: wheres+=" or"
        wheres += f" `FIN`.`{p}` <> ''"
    if wheres == "":
        wheres = " True"
    if iddp:
        if type(iddp) is list:
            iddp = ",".join([str(i) for i in iddp])
            datapoints = f"""WHERE `deep4chem`.`Datapoints`.`idDatapoints` IN ({iddp})"""
        else:
            datapoints = f"""WHERE `deep4chem`.`Datapoints`.`idDatapoints` = {iddp}"""
    elif DOI:
        datapoints = f"""WHERE `deep4chem`.`Datapoints`.`idRefers` IN 
                        (SELECT idRefers FROM `deep4chem`.`Refers` WHERE `deep4chem`.`Refers`.`DOI` = '{DOI}')"""
    else:
        datapoints= ""
    SQL = SQL.format(select_column,brackets,joins,wheres,datapoints)
    return SQL


# In[ ]:


def bring_DB():
    import pymysql
    connection = pymysql.connect(
            host='163.152.42.111',
            port=3306,
            user='root',
            passwd='ufslab223',
            db='deep4chem',
            cursorclass = pymysql.cursors.DictCursor
            )
    
    SQL =makeQuery_DBselect(properties=['Abs',"FWHM_Abs",'log_extinc',"Emi","FWHM_Emi","PLQY","lifetime","HOMO","HOMO_Method","LUMO","LUMO_Method","Optical_bandgap"])
    with connection.cursor() as cursor:
        cursor.execute(SQL,)
        result = cursor.fetchall()
    
    
    connection.close()
    result= pd.DataFrame(result)
    return result


# In[ ]:


def delete_weird_mol(data_table, column):
    error_mols=[]
    for i, smiles in enumerate(list(data_table[column])):
        try:
            mol = Chem.MolFromSmiles(smiles,  sanitize = True)
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f'{i} 번째 분자 이상함')
            print(e)
            error_mols.append(i)

    data_table= data_table.drop(error_mols)
    data_table.index=range(len(data_table))
    return data_table


def delete_weird_mol_and_None_except_Nan(data_table, column):
    # 인덱스 기반으로 삭제할거니까 인덱스 재정렬 추가
    data_table.reset_index(drop=True, inplace = True)
    error_mols=[]
    for i, smiles in enumerate(list(data_table[column])):
        if pd.isna(smiles):  # 결측치 Nan
            continue
        try:
            mol = Chem.MolFromSmiles(smiles,  sanitize = True) # sanitize 에서 문법오류 제거
            if mol is None:
                print(f'{i}번째 분자 mol=None (smiles가 아닌 값)')
                error_mols.append(i)
                continue
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f'{i} 번째 분자 이상하고 결측치도 아님')
            print(e)
            error_mols.append(i)

    data_table= data_table.drop(error_mols)
    data_table.index=range(len(data_table))
    return data_table

# In[ ]:


def delete_metal_mol(data_table, column):
    data_table.reset_index(drop=True, inplace = True)
    metal_atomic_numbers = [
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        87, 88, 89, 90, 91, 92, 93, 94
    ]
    
    def contains_metal(smiles):
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in metal_atomic_numbers:
                return True
        return False
    
    metal_list=[]
    for idx, smiles in enumerate(list(data_table[column])):
        if contains_metal(smiles)==True:
            metal_list.append(idx)
    
    data_table = data_table.drop(metal_list)
    data_table.index=range(len(data_table))
    return data_table

def delete_metal_mol_except_Nan(data_table, column):
    metal_atomic_numbers = [
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        87, 88, 89, 90, 91, 92, 93, 94
    ]
    
    def contains_metal(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'너 뭐야!!!!!!!!!!!!!!!!!{smiles}')
        except:
            print(f'너 뭐야22!!!!!!!!!!!!!!!!!{smiles}')

        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in metal_atomic_numbers:
                return True
        return False


    
    metal_list=[]
    for idx, smiles in enumerate(list(data_table[column])):
        if pd.isna(smiles):
            continue
        
        if contains_metal(smiles)==True:
            print(f'{idx}번째 분자 금속원소 포함됨')
            metal_list.append(idx)

    
    data_table = data_table.drop(metal_list)
    data_table.index=range(len(data_table))
    return data_table