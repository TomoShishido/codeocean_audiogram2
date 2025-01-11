import cv2, os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def yolo2df(YOLOTXTFILEDIR): #YOLOTXTFILEDIR = dir/*.txt
    df = pd.DataFrame()
    files = glob.glob(YOLOTXTFILEDIR)
    #To skip .txt files
    for file in files:
        if file.endswith('txt'):
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
              
            with open(file) as f:
                txtlines = f.readlines()
            #draw a frame
            for textline in txtlines:
                target_info = textline.split() #target_info =[label, x, y, w, h]
                if target_info[0] == '0':# in the case of a0
                    item = pd.Series([namewithoutext, 0, float(target_info[1]), float(target_info[2])],index=['filename', 'label', 'x', 'y'])
                elif target_info[0] == '1':# in the case of a1
                    item = pd.Series([namewithoutext, 1, float(target_info[1]), float(target_info[2])],index=['filename', 'label', 'x', 'y'])
                elif target_info[0] == '2':# in the case of a2
                    item = pd.Series([namewithoutext, 2, float(target_info[1]), float(target_info[2])],index=['filename', 'label', 'x', 'y'])
                elif target_info[0] == '3':# in the case of a3
                    item = pd.Series([namewithoutext, 3, float(target_info[1]), float(target_info[2])],index=['filename', 'label', 'x', 'y'])
                elif target_info[0] == '4':# in the case of a4
                    item = pd.Series([namewithoutext, 4, float(target_info[1]), float(target_info[2])],index=['filename', 'label', 'x', 'y'])
                else:
                    pass
                df=df.append([item],ignore_index=True)
                # df=pd.concat([df, item])
    return df

''' 
confusion matrix
TP, TN, FP
recall, precision, f1 score

'''
def check_match_item(ref_row, detected_df, tolerance=0.02):
    matches = []
    for _, det_row in detected_df.iterrows():
        if ref_row['filename'] == det_row['filename'] and ref_row['label'] == det_row['label']:
            x_diff = abs(ref_row['x'] - det_row['x'])
            y_diff = abs(ref_row['y'] - det_row['y'])
            within_tolerance = x_diff <= tolerance and y_diff <= tolerance
            if within_tolerance:
                matches.append({
                    'detected_idx': det_row.name,
                    'detected_x': det_row['x'],
                    'detected_y': det_row['y'],
                    'x_diff': x_diff,
                    'y_diff': y_diff,
                    'within_tolerance': within_tolerance
                })
    return matches

def check_match_dfs(ref_df, detected_df):
    results = []
    for _, ref_row in ref_df.iterrows():
        matches = check_match_item(ref_row, detected_df)
        if matches:
            results.append({
                'ref_idx': ref_row.name,
                'ref_filename': ref_row['filename'],
                'ref_label': ref_row['label'],
                'ref_x': ref_row['x'],
                'ref_y': ref_row['y'],
                'matches': matches
            })
    # Display results
    for result in results:
        print(f"Reference row index {result['ref_idx']} ({result['ref_filename']}, {result['ref_label']}):")
        for match in result['matches']:
            print(f"  -> Detected row index {match['detected_idx']} (x={match['detected_x']}, y={match['detected_y']}): "
                f"x_diff={match['x_diff']:.3f}, y_diff={match['y_diff']:.3f}, within_tolerance={match['within_tolerance']}")
        if not result['matches']:
            print("  -> No match found")
        print()
    return results

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=40)
    plt.yticks(tick_marks, classes,fontsize=40)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=60)

    plt.tight_layout()
    plt.ylabel('True',fontsize=40)
    plt.xlabel('Predicted',fontsize=40)
    plt.show()

def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    recall = tp/(tp +fn)
    precision = tp/(tp+fp)
    # f1score = 2*(precision*recall)/(precision + recall)
    EDR = 100*(fp+fn)/tp
    if print_res == 1:
        print('Recall (Sensitivity) =  {:.2f}'.format(tp/(tp+fn)))
        # print('Specificity =     {:.2f}'.format(tn/(tn+fp)))
        # print('Accuracy =     {:.2f}'.format((tp+tn)/(tp+fp+tn+fn)))
        print('Precision(PPV) =     {:.2f}'.format(tp/(tp+fp)))
        # print('NPV =     {:.2f}'.format(tn/(tn+fn)))
        # print('F1score = {:.2f}'.format(f1score))
        print('EDR = {:.1f}'.format(EDR))
        
    return recall, precision, EDR

def set_cm(tn,fp,fn,tp):
    cm=np.matrix([[0,0],[0,0]])
    cm[1,1]=tp
    cm[1,0]=fn
    cm[0,1]=fp
    cm[0,0]=tn
    return cm

def analysis2confusionmatrix(ref_df, detected_df,title):
    TN = FP = FN = TP =0
    ref_count = ref_df.shape[0]
    detected_count = detected_df.shape[0]
    FNTPresult = check_match_dfs(ref_df, detected_df)
    TP = len(FNTPresult)
    FN = ref_count - TP
    FPresult = check_match_dfs(detected_df, ref_df)
    FP = detected_count - len(FPresult)

    cm = set_cm(0,FP,FN,TP)
    plot_confusion_matrix(cm, ['0', '1'], title=title)
    recall, precision, EDR = show_data(cm, print_res = 1)