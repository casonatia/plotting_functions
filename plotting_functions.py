import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def custom_font() :
    font = {'weight' : 'bold',
            'size'   : 20}
    matplotlib.rc('font', **font)

def clear_font() :
    font = {'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)

def color_picker( n_colors, cm = 'Spectral', return_rgb = True, return_cmap = False) :
    ''' 
    Data una colormap (discreta o continua) di matplotlib ne estrae n_colori.
    I colori estratti vengono restituiti sotto forma di array o di colormap
    nota : i colori sono normalizzati 
    '''
    cmap = plt.cm.get_cmap(cm, n_colors)
    rgb_colors = cmap(np.linspace(0, 1, n_colors))
    if return_rgb == True and return_cmap == False :
        return rgb_colors
    elif return_cmap == True and return_rgb == False:
        return ListedColormap(rgb_colors)
    elif return_cmap == True and return_rgb == True:
        d = {}
        d['rgb'] = rgb_colors
        d['cmap'] = cmap
        return d
    else :
        print('No object returned')


class classification :
    def __init__(self, df = None, ytrue = None, proba_cols = None, ypred = None) :
        self.df = df
        self.ytrue = ytrue
        self.ypred = ypred
        self.proba_cols = proba_cols

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        cbar.remove()

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="Black", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                        textcolors=["black", "white"],
                        threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A list or array of two color specifications.  The first is used for
            values below a threshold, the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts


    def round_up(self, df, parts) :
        division = np.linspace(0,1,parts)
        conditions = {}
        for el in range(0, len(division) - 1) :
            conditions['c{}'.format(el)] =  (df.perc >= division[el]) & (df.perc < division[el + 1])
        vals = division[1:]
        df['perc'] = np.select(list(conditions.values()), vals)
        return df.perc.values.tolist()



    def gain_plot(self, parts = 10, subsample = None, savedir = None, savename = None, pdf = None) :
        custom_font()
        df = self.df
        ytrue = self.ytrue
        proba_cols = self.proba_cols
        if subsample :
            df = df.sample(subsample)
        rgbs = color_picker( len(proba_cols), cm = 'Spectral', return_rgb = True, return_cmap = False)
        color_dict = dict(zip(proba_cols, rgbs.tolist()))
        x = np.linspace(0,1,parts)
        random_choice_y = x
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8), sharex = False)
        fig.suptitle('Gain plot with ratio')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Percentage of Samples')
        ax1.set_ylabel('Gain')
        ax2.set_aspect('equal')
        ax2.set_xlabel('Percentage of Samples')
        ax2.set_ylabel('Ratio')
        ax1.plot(x, random_choice_y)
        for col in proba_cols :
            d = df[[col, ytrue]].sort_values(col, ascending = False).reset_index(drop = True)\
                .reset_index(drop = False).rename(columns = {'index':'cumulative'})
            d[ytrue] = np.where(d[ytrue] == col, 1, 0)
            d['perc'] = d.cumulative / df.shape[0]
            d['perc'] = self.round_up(d, parts)
            d = d.groupby(['perc']).agg({ytrue : 'sum', 'cumulative' : 'count'}).reset_index(drop = False)
            d['cumsum_'] = d[ytrue].cumsum()
            d['cumsum_'] = d.cumsum_ / max(d.cumsum_)
            d['ratio'] =  d[ytrue] / d.cumulative
            value = d.cumsum_.values.tolist()
            ratio = d.ratio.values.tolist()
            value.insert(0, 0)
            ratio.insert(0, 0)
            ax1.plot(x, value, label = col, linewidth = 3, c = color_dict[col])
            ax2.plot(x, ratio, label = col, linewidth = 3, c = color_dict[col])
        ax1.grid()   
        ax2.grid() 
        plt.legend()
        plt.tight_layout()
        plt.draw()
        if savedir :
            if savename :
                name = os.path.join(savedir,savename)
                plt.savefig(name)
            else : 
                print('please, specify a savename')
        if pdf :
            pdf.savefig()
        clear_font()
        

    def liftchart(self, parts = 10, plot_type = 'all', subsample = None, savedir = None, savename = None, pdf = None) :
        # if plot_type == 'all' plot all values inside the same chart, else 
        # make one chart for each target
        custom_font()
        df = self.df
        ytrue = self.ytrue
        proba_cols = self.proba_cols
        if subsample :
            df = df.sample(subsample)
        if plot_type == 'all' :
            x = np.linspace(0,1,parts)
            plt.figure(figsize = (16,9))
            for col in proba_cols :
                d = df[[col, ytrue]].sort_values(col, ascending = False).reset_index(drop = True)\
                    .reset_index(drop = False).rename(columns = {'index':'cumulative'})
                d[ytrue] = np.where(d[ytrue] == col, 1, 0)
                standard_freq = d[ytrue].sum() / d.shape[0]
                d['perc'] = d.cumulative / df.shape[0]
                d['perc'] = self.round_up(d, parts)
                d = d.groupby(['perc']).agg({ytrue : 'sum', 'cumulative' : 'max'}).reset_index(drop = False)
                d['true_cum'] = d[ytrue].cumsum()
                d['frac'] = d['true_cum'] / d.cumulative
                d['lift'] =  d.frac / standard_freq              
                values = d.lift.values.tolist()
                values.insert(0, d.lift.loc[0])
                plt.plot(x, values, label = col, linewidth = 4)
            plt.grid()
            plt.legend(loc = 'upper right')
            plt.title('Lift Chart')
            plt.xlabel('Sample size')
            plt.ylabel('Lift')
            plt.tight_layout()
            if savedir :
                if savename :
                    name = os.path.join(savedir,savename)
                    plt.savefig(name)
                else : 
                    print('please, specify a savename')
            if pdf :
                pdf.savefig()
        elif plot_type == 'single' :
            x = np.linspace(0,1,parts)
            for col in proba_cols :
                d = df[[col, ytrue]].sort_values(col, ascending = False).reset_index(drop = True)\
                    .reset_index(drop = False).rename(columns = {'index':'cumulative'})
                d[ytrue] = np.where(d[ytrue] == col, 1, 0)
                standard_freq = d[ytrue].sum() / d.shape[0]
                d['perc'] = d.cumulative / df.shape[0]
                d['perc'] = self.round_up(d, parts)
                d = d.groupby(['perc']).agg({ytrue : 'sum', 'cumulative' : 'max'}).reset_index(drop = False)
                d['true_cum'] = d[ytrue].cumsum()
                d['frac'] = d['true_cum'] / d.cumulative
                d['lift'] =  d.frac / standard_freq  
                values = d.lift.values.tolist()
                values.insert(0, d.lift.loc[0])
                plt.figure(figsize = (16,9))
                plt.plot(x, values, linewidth = 4, label = col)
                plt.grid()
                plt.legend()
                plt.title('Lift Chart {}'.format(col))
                plt.xlabel('Sample size')
                plt.ylabel('Lift')
                plt.tight_layout()
                if savedir :
                    name = os.path.join(savedir,'{}.png'.format(col))
                    plt.savefig(name)
                if pdf :
                    pdf.savefig()
        clear_font()


    def roc_plot(self, plot_type = None, subsample = None, savedir = None, savename = None, pdf = None) :
        custom_font()
        df = self.df
        ytrue = self.ytrue
        proba_cols = self.proba_cols
        if subsample :
            df = df.sample(subsample)
        x = np.arange(0, 1.1, 0.1) 
        random_choice_y = x
        if plot_type == 'single' :
            for col in proba_cols :
                d = df.copy()
                d['pred'] = np.where(d[ytrue] == col, 1, 0)
                d = d[[col, 'pred']] 
                fpr, tpr, thresholds = metrics.roc_curve(d.pred, d[col], pos_label= 1)
                auc = round(metrics.roc_auc_score(d.pred, d[col]), 3)
                plt.figure(figsize = (10,10))
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid()
                plt.plot(fpr, tpr, label = col, linewidth = 3)
                plt.plot(x, random_choice_y, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve {} (AUC : {})'.format(col, auc))
                plt.tight_layout()
                plt.draw()
                if savedir :
                    name = os.path.join(savedir,'{}.png'.format(col))
                    plt.savefig(name)
                if pdf :
                    pdf.savefig()
        elif plot_type == 'all':
            plt.figure(figsize = (10,10))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.plot(x, random_choice_y, linestyle='--', label = 'random')   
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            for col in proba_cols :
                d = df.copy()
                d['pred'] = np.where(d[ytrue] == col, 1, 0)
                d = d[[col, 'pred']] 
                fpr, tpr, thresholds = metrics.roc_curve(d.pred, d[col], pos_label= 1)
                auc = round(metrics.roc_auc_score(d.pred, d[col]), 3)
                plt.plot(fpr, tpr, linewidth = 3, label = '{}(AUC:{})'.format(col, auc) )
            plt.legend()
            plt.tight_layout()
            plt.draw()
            if savedir :
                if savename :
                    name = os.path.join(savedir,savename)
                    plt.savefig(name)
                else : 
                    print('please, specify a savename')
            if pdf :
                pdf.savefig()   
        else :
            'Please specify a plot type'
        clear_font()


    def confusion_matrix_plot(self, normalize = 'true', only_matrix = False, labels = None, figsize = None, savedir = None, savename = None, pdf = None) :
        custom_font()
        df = self.df 
        ytrue = self.ytrue
        ypred = self.ypred
        if labels is None : 
            labels = df[ytrue].value_counts().index.tolist()
        if normalize not in ['true', 'pred', 'all', None] :
            print('normalize must be in  ["true", "pred", "all", None]')
        if only_matrix :
            cm = confusion_matrix(df[ytrue], df[ypred], labels=None, sample_weight=None, normalize=normalize)
            if figsize :
                fig, ax = plt.subplots(figsize = figsize)
            else :
                fig, ax = plt.subplots(figsize = (10,10))
            im, cbar = self.heatmap(cm, labels, labels, ax=ax, cmap="Blues")
            texts = self.annotate_heatmap(im, valfmt="{x:.3f}")
            fig.tight_layout()
            if normalize is None :
                plt.title('Confusion Matrix')
            else :
                plt.title('Normalized Confusion Matrix')
        else :   
            cm = confusion_matrix(df[ytrue], df[ypred], labels=None, sample_weight=None, normalize=normalize)
            if figsize :
                fig, axs = plt.subplots(2, 2, figsize=figsize, sharey= False)
            else :
                fig, axs = plt.subplots(2, 2, figsize=(12,12), sharey= False)
            if normalize is None :
                fig.suptitle('Confusion Matrix')
            else :
                fig.suptitle('Normalized Confusion Matrix')
            im, cbar = self.heatmap(cm, labels, labels, ax=axs[0,0], cmap="Blues", cbarlabel="Normalized Confusion matrix")
            texts = self.annotate_heatmap(im, valfmt="{x:.3f}")
            cm_c = confusion_matrix(df[ytrue], df[ypred], labels=None, sample_weight=None, normalize=None)
            axs[0,1].barh(labels, [sum(i) for i in cm_c])
            axs[0,1].axes.get_xaxis().set_visible(False)
            axs[0,1].axes.get_yaxis().set_visible(False)
            axs[0,1].invert_yaxis()
            axs[0,1].axis('off')
            axs[1,0].bar(labels, [sum(cm_c[:,i]) for i in range(0, cm_c.shape[1])])
            axs[1,0].axes.get_xaxis().set_visible(False)
            axs[1,0].axes.get_yaxis().set_visible(False)
            axs[1,0].invert_yaxis()
            axs[1,0].axis('off')
            fig.delaxes(axs.flatten()[3])
            plt.subplots_adjust(wspace=0.0001, hspace=0.0001)
        if savedir :
            if savename :
                name = os.path.join(savedir,savename)
                plt.savefig(name, bbox_inches = "tight")
            else : 
                print('please, specify a savename')
        if pdf :
            pdf.savefig(bbox_inches = "tight")
        plt.show()
        clear_font()

    def plot_classification_report(self, figsize = None, savedir = None, savename = None, pdf = None ) :
        custom_font()
        df = self.df
        ytrue = self.ytrue
        ypred = self.ypred
        report = classification_report(df[ytrue].values, df[ypred].values, output_dict = True)
        accuracy = np.round(report['accuracy'], 3)
        del report['accuracy']
        xlabs = list(report.keys())
        ylabs = list(report[xlabs[0]].keys())
        matrix = np.zeros((len(xlabs), len(ylabs)))
        for idx in range(0, len(xlabs)) :
            for idy in range(0, len(ylabs)) :
                matrix[idx][idy] = report[xlabs[idx]][ylabs[idy]]
        if figsize :
            fig, ax = plt.subplots(figsize = figsize)
        else :
            fig, ax = plt.subplots(figsize = (15,15))
        im, cbar = self.heatmap(matrix, xlabs, ylabs, ax = ax, cmap = 'Blues', vmin = np.min(matrix[:,0:-1]), vmax = np.max(matrix[:,0:-1]))
        texts = self.annotate_heatmap(im, valfmt="{x:.3f}")
        plt.title('Classification Report\nAccuracy:{}'.format(accuracy))
        fig.tight_layout()
        if savedir :
            if savename :
                name = os.path.join(savedir,savename)
                plt.savefig(name)
            else : 
                print('please, specify a savename')
        if pdf :
            pdf.savefig()
        plt.show()
        clear_font()


class evaluation :
    def plot_bivariate_multi(data, feature, binary_score, hist_bins=False, max_show_bins=False, max_feature_value=False, show_plot=False, output_dir=None):
        '''
        Plots bivariate chart of binned features against average score in bins;
        it detects whether it's quantitative or categorical feature and treats it consequently;
        use hist_bins and max_feature_value for quantitatives, max_show_bins for categoricals.
        You can pass more than one binary_score in the following format : [[binary_score1, 'color1'], [binary_score2, 'color2'], ...]
        '''
        custom_font()
        # FOR CATEGORICAL PREDICTORS
        if data.dtypes[data.columns == feature].values[0] == 'O':
            vc = data[feature].value_counts()
            hist = np.array(vc.values)
            bins = np.array(vc.index)
            if max_show_bins:
                hist, bins,  = hist[0:max_show_bins], bins[0:max_show_bins]
            
            # print(bins)
            fig, ax1 = plt.subplots(figsize = (12,6))
            color = '#00bfff'
            ax1.set_xlabel(feature)
            if len(bins) > 15 or np.mean([len(x) for x in bins]) > 10:
                ax1.set_xticklabels(bins, rotation= 90)
            ax1.set_ylabel('Value Count', color='#2A2B33')
            ax1.bar(x = bins, height = hist, color = color, label = 'Value Count', alpha = 0.6)
            ax1.tick_params(axis='y', labelcolor='#2A2B33')
        
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
            ax2.set_ylabel('Average Score', color='#682FAF')  # we already handled the x-label with ax1
            
            colors_ = color_picker( len(binary_score), cm = 'Spectral', return_rgb = True)
            for bs in range(0, len(binary_score)):
                binned_binary = [np.mean(data.loc[data[feature] == bins[b], binary_score[bs][0]]) for b in range(len(bins))]
                if len(binary_score[bs]) > 1 :
                    color = binary_score[bs][1]
                else :
                    color = colors[bs]
                ax2.plot(bins, binned_binary, color = color, marker ='o', label = (bs[0] + '_mean'))

            ax2.tick_params(axis='y', labelcolor='#682FAF')
        
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            plt.title('Scores distribution across different {}'.format(feature))

        # FOR QUANTITATIVE PREDICTORS
        else:
            if max_feature_value:
                data = data.loc[data[feature] <= max_feature_value, :].copy()
                
            hist, bins = np.histogram(data[feature].dropna().values)
            if hist_bins:
                hist, bins = np.histogram(data[feature].dropna().values, bins=hist_bins)
            
            while len(bins) > len(hist):
                hist = np.append(hist, 0)
        
            
            bl = np.abs(bins[1]) - np.abs(bins[0])
            w = bl * 0.65
            # print(bins)
            fig, ax1 = plt.subplots(figsize = (12,6))
            color = '#00bfff'
            ax1.set_xlabel(feature)
            ax1.set_ylabel('Value Count', color='#2A2B33')
            ax1.bar(x = bins, height = hist, color = color, label = 'Value Count', width = w, alpha = 0.6)
            ax1.tick_params(axis='y', labelcolor='#2A2B33')
        
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
            ax2.set_ylabel('Average Score', color='#682FAF')  # we already handled the x-label with ax1
            colors_ = color_picker( len(binary_score), cm = 'Spectral', return_rgb = True)
            for bs in range(0, len(binary_score)):
                binned_binary = [np.mean(data.loc[(data[feature] < bins[b+1]) &
                                                (data[feature] >= bins[b]),
                                                binary_score[bs][0]])
                                for b in range(len(bins)-1)]
        
                
                while len(bins) > len(binned_binary):
                    binned_binary = np.append(binned_binary, 0)  
                if len(binary_score[bs]) > 1 :
                    color = binary_score[bs][1]
                else :
                    color =  colors_[bs]

                ax2.plot(bins, binned_binary, color = color, marker ='o', label = (bs[0] + '_mean'))

            ax2.tick_params(axis='y', labelcolor='#682FAF')
        
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            plt.title('Scores distribution across different {}'.format(feature))
            
        if output_dir != None:
            plt.savefig(os.path.join(output_dir, 'Bivariate chart for {}.png'.format(feature)), bbox_inches = "tight")
        if show_plot:
            plt.show()
        clear_font()