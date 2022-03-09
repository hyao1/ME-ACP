# ME-ACP
The offical code of paper :ME-ACP: Multi-view Neural Networks with Ensemble Model for Identification of Anticancer Peptides.  

cross_val.py is utilized for ACP740 and ACP240 datasets, and indenpendent.py is for Main and Alternate dataset  

1、feature extraction  
   >peptide level features should be extracted by http://bioinformatics.hitsz.edu.cn/BioSeq-Analysis/download.  
   >resdual level features should be extracted by the tool in the folder named "Residual level".  
   >please rename all xxx-DT.csv into xxx-DT1.csv  
  
2、put the files from above into a folder as following form：  
   >ACP740: Anti-cancer-data/ACP-740-240-feature/ACP-740-feature/xxx.csv  
   >ACP240: Anti-cancer-data/ACP-740-240-feature/ACP-240-feature/xxx.csv  
   >Main: Anti-cancer-data/Main/xxx.csv  
   >Alternate: Anti-cancer-data/Alternate/xxx.csv  
    
3、train your model using command:  
   >ACP740: python cross_val.py --data_name ACP740 --begin 3  
   >ACP240: python cross_val.py --data_name ACP240 --begin 3  
   >Main: python indenpendent.py --data_name Main --begin 3  
   >Alternate: python indenpendent.py --data_name Alternate --begin 3  
            
                     
