import sys
import pandas as pd
import numpy as np
from app import MlProject
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QSplitter, QGroupBox,
    QScrollArea, QLabel, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtGui import QPainter

def impute_missing_vals(input_df, proj: MlProject = None):
    df = input_df.copy()
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    
    df_num = df[[c for c in df.columns if not proj.columns[c].isDiscrete()]]
    df_cat = df[[c for c in df.columns if proj.columns[c].isDiscrete()]]
    
    df_num_imputed = pd.DataFrame(num_imputer.fit_transform(df_num), columns=df_num.columns)
    df_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(df_cat), columns=df_cat.columns)
    
    df.update(df_num_imputed)
    df.update(df_cat_imputed)
    return df

def remove_invariants(input_df, proj: MlProject = None, threshold=0.0):
    df = input_df.copy()
    target = proj.targetVariable

    target_column = df[target]
    feature_columns = df.drop(columns=[target])
    for column in [c for c in feature_columns.columns if feature_columns[c].dtype == 'object']:
            le = LabelEncoder()
            feature_columns[column] = le.fit_transform(feature_columns[column])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(feature_columns)

    columns_kept = list(feature_columns.columns[selector.get_support()])
    to_drop = [c for c in feature_columns.columns if c not in columns_kept]
    df = df.drop(columns=to_drop)
    return df

def handle_outliers(input_df, proj: MlProject = None):
    df = input_df.copy()
    target = proj.targetVariable

    df_num = df[[c for c in df.columns if not proj.columns[c].isDiscrete() and c != target]]
    
    z_scores = np.abs(stats.zscore(df_num))
    for col in df_num.columns:
        df[col] = np.where(z_scores[col] > 3, df[col].median(), df[col])
    return df

def remove_lin_dep(input_df, proj: MlProject = None, vif_thresh=5):
    df = input_df.copy()
    target = proj.targetVariable

    feature_columns = df.drop(columns=[target])
    for column in [c for c in feature_columns.columns if feature_columns[c].dtype == 'object']:
            le = LabelEncoder()
            feature_columns[column] = le.fit_transform(feature_columns[column])

    vif_data = pd.DataFrame({'feature':feature_columns.columns,
                             'VIF':[variance_inflation_factor(feature_columns.values, i) for i in range(len(feature_columns.columns))]})

    to_drop = vif_data[vif_data['VIF'] > 5]['feature']
    df = df.drop(columns=to_drop)
    return df

def encode_vars(input_df, proj: MlProject = None, onehot=True):
    df = input_df.copy()
    target = proj.targetVariable
    if df[target].dtype == 'object':
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

    df_cat = df[[c for c in df.columns if proj.columns[c].isDiscrete() and c != target]]
    if onehot:
        df = pd.get_dummies(df, columns=df_cat.columns, drop_first=True)
    else:
        for column in df_cat.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df_cat[column])
    return df

def pre_process(df, params=None, proj: MlProject = None):
    if params is None:
        params = {'impute':True, 'remove_invariants':True, 'handle_outliers':True, 'vif_threshold':5, 'encoding':'onehot'}

    if params['impute']:
        df = impute_missing_vals(df, proj)
    
    if params['remove_invariants']:
        df = remove_invariants(df, proj)

    if params['handle_outliers']:
        df = handle_outliers(df, proj)
    
    if params['vif_threshold'] > 0:
        df = remove_lin_dep(df, proj, vif_thresh=params['vif_threshold'])

    encoded_df = encode_vars(df, proj, onehot=(params['encoding'] == 'onehot'))
    return encoded_df, df

def training_and_evaluation(df, proj: MlProject = None):

    target = proj.targetVariable
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers and regressors
    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025, probability=True, random_state=42),
        "RBF SVM": SVC(gamma=2, C=1, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42
        ),
        #"Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=42),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Gaussian Process Regression": GaussianProcessRegressor(1.0 * RBF(1.0)),
        "Decision Tree Regression": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Neural Net Regression": MLPRegressor(
            hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, random_state=42
        ),
        "AdaBoost Regression": AdaBoostRegressor(
            n_estimators=50, learning_rate=1.0, random_state=42
        ),
        "K Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
    }

    # Determine whether the task is classification or regression
    if proj.columns[target].isDiscrete():
        task_type = 'classification'
        models = classifiers
        metrics = {
            "Accuracy": accuracy_score,
            "F1 Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            "ROC AUC": lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovo')
        }
        scoring = 'accuracy'  # Scoring for cross-validation in classification
    else:
        task_type = 'regression'
        models = regressors
        metrics = {
            "Mean Absolute Error": mean_absolute_error,
            "Mean Squared Error": mean_squared_error,
            "Root Mean Squared Error": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "R-squared": r2_score
        }
        scoring = 'neg_mean_squared_error'  # Scoring for cross-validation in regression

    # Initialize results storage
    results = []
    perm_importance_dict = {}
    confusion_matrices = {}

    # Evaluate all models
    for name, model in models.items():
        print(f"Training and evaluating model: {name}")
        model = make_pipeline(StandardScaler(), model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        result = {"Model": name}
        for metric_name, metric_func in metrics.items():
            if metric_name == "ROC AUC" and task_type == 'classification' and len(y.unique()) == 2:
                score = metric_func(y_test, model.predict_proba(X_test)[:, 1])
            else:
                score = metric_func(y_test, y_pred)
            result[metric_name] = score

        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        if task_type == 'classification':
            result["Cross-validated Accuracy"] = cv_scores.mean()
        else:
            result["Cross-validated RMSE"] = np.sqrt(-cv_scores.mean())

        results.append(result)

        # Confusion matrix for classification
        if task_type == 'classification':
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm

        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean
        })

        # Combine importances for one-hot encoded features
        original_features = {}
        for feature in perm_importance_df['feature']:
            base_feature = feature.split('_')[0]
            if base_feature in original_features:
                original_features[base_feature] += perm_importance_df.loc[perm_importance_df['feature'] == feature, 'importance'].values[0]
            else:
                original_features[base_feature] = perm_importance_df.loc[perm_importance_df['feature'] == feature, 'importance'].values[0]

        perm_importance_df_combined = pd.DataFrame(list(original_features.items()), columns=['feature', 'importance'])
        perm_importance_df_combined = perm_importance_df_combined.sort_values(by='importance', ascending=False)
        perm_importance_dict[name] = perm_importance_df_combined

    # Convert results to a DataFrame and rank them by performance
    results_df = pd.DataFrame(results)

    if task_type == 'classification':
        best_model_idx = results_df['Accuracy'].idxmax()
        results_df.sort_values(by="Accuracy", ascending=False, inplace=True)
    else:
        best_model_idx = results_df['R-squared'].idxmax()
        results_df.sort_values(by="R-squared", ascending=False, inplace=True)
    
    output = {
        'results_df': results_df,
        'best_model_idx': best_model_idx,
        'perm_importance_dict': perm_importance_dict,
        'confusion_matrices': confusion_matrices,
    }
    return output

class ZoomableView(QGraphicsView):
    def __init__(self, widget):
        scene = QGraphicsScene()
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # Add the widget to the scene
        proxy_widget = QGraphicsProxyWidget()
        proxy_widget.setWidget(widget)
        scene.addItem(proxy_widget)

    def wheelEvent(self, event):
        # Check if the Control key is pressed
        if event.modifiers() == Qt.ControlModifier:
            # Zoom in or out depending on the scroll direction
            zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8
            self.scale(zoom_factor, zoom_factor)
        else:
            # Default scrolling behavior
            super().wheelEvent(event)


class ModelEvaluationApp(QMainWindow):
    def __init__(self, support):
        super().__init__()
        self.setWindowTitle("Model Evaluation Results")
        self.setGeometry(100, 100, 1000, 800)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create vertical splitter for the first 3 sub-windows
        vertical_splitter = QSplitter(Qt.Vertical)
        sub_window_heights = [200, 100, 300]  # Ratios 2:1:3 for the top three windows

        sub_horizontal_splitter = QSplitter(Qt.Horizontal)
        sub_horizontal_splitter.addWidget(self.create_methods_subwindow())
        sub_horizontal_splitter.addWidget(self.create_results_subwindow(results_df=support['results_df'], best_model_idx=support['best_model_idx']))

        # Add sub-windows with content from ModelEvaluationApp
        # vertical_splitter.addWidget(self.create_methods_subwindow())
        vertical_splitter.addWidget(sub_horizontal_splitter)
        vertical_splitter.addWidget(self.create_pipeline_subwindow(params=support['params']))
        # vertical_splitter.addWidget(self.create_results_subwindow(results_df=support['results_df'], best_model_idx=support['best_model_idx']))

        # Set initial sizes of the three vertical windows
        vertical_splitter.setSizes(sub_window_heights)

        # Create horizontal splitter for the last 2 sub-windows
        horizontal_splitter = QSplitter(Qt.Horizontal)

        # Add sub-windows with content from ModelEvaluationApp
        horizontal_splitter.addWidget(self.create_importance_subwindow(perm_importance_dict=support['perm_importance_dict']))
        horizontal_splitter.addWidget(self.create_confusion_matrix_subwindow(confusion_matrices=support['confusion_matrices']))

        # Create main splitter to combine vertical and horizontal splitters
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(vertical_splitter)
        main_splitter.addWidget(horizontal_splitter)

        # Set initial sizes for the main splitter to achieve 2:1:3:4 ratio
        main_splitter.setSizes([600, 600])  # Ratios 6:4 for the top three and bottom two combined

        # Add main splitter to the layout
        main_layout.addWidget(main_splitter)

        # Set main widget as central widget
        self.setCentralWidget(main_widget)

    def create_scrollable_subwindow(self, title, content_widget, init_zoom_out_factor=1):
        group_box = QGroupBox(title)
        layout = QVBoxLayout(group_box)
        zoom_view = ZoomableView(content_widget)
        zoom_view.scale(init_zoom_out_factor, init_zoom_out_factor)
        layout.addWidget(zoom_view)
        return group_box

    def create_methods_subwindow(self):
        methods_widget = QWidget()
        methods_layout = QVBoxLayout(methods_widget)
        methods_text = ("<b>Preprocessing:</b> Handling missing values, removing invariant features, handling outliers, "
                        "checking for multicollinearity using VIF, and encoding categorical variables.<br>"
                        "<b>Training:</b> Various classifiers/regressors trained based on task type.<br>"
                        "<b>Evaluation:</b> Metrics include accuracy, F1 score, ROC AUC (classification) "
                        "or MAE, RMSE, R-squared (regression). Cross-validation ensures model stability.<br>"
                        "<b>Permutation Importance:</b> Assesses feature importances by evaluating error increase with permuted features.<br>"
                        "<b>Confusion Matrix:</b> Provides insight into classifier error types.")

        methods_label = QLabel(methods_text)
        methods_label.setWordWrap(True)
        methods_label.setStyleSheet("font-size: 10pt;")
        methods_layout.addWidget(methods_label)

        # return methods_widget
        return self.create_scrollable_subwindow("Methods Used", methods_widget)

    def create_pipeline_subwindow(self, params):
        pipeline_widget = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_widget)

        # Create the figure and axis
        pipeline_fig, ax = plt.subplots(figsize=(10.5, 1))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)

        # Define the stages
        stages = ['Data', 'Impute Missing', 'Remove Invariants',
                'Handle Outliers', 'Reduce Colinearity', 'Encoding', 'Training', 'Evaluation']

        # Store box information
        box_info = []

        # Add text boxes to calculate their widths
        for stage in stages:
            text = ax.text(0, 0.5, stage, ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='skyblue', alpha=0.5, edgecolor='blue'))
            # Get the bounding box of the text in display coordinates
            bbox = text.get_window_extent(renderer=pipeline_fig.canvas.get_renderer())
            # Transform bounding box to data coordinates
            bbox_data = bbox.transformed(ax.transData.inverted())
            box_info.append(bbox_data.width)
            text.remove()  # Remove the temporary text

        # Calculate total width and spacing
        total_text_width = sum(box_info)
        desired_gap = 0.25
        total_gap_width = desired_gap * (len(stages) - 1)
        total_width = total_text_width + total_gap_width

        # Calculate starting position
        x_start = (ax.get_xlim()[1] - total_width) / 2

        # Add the text boxes at calculated positions
        current_x = x_start
        for i, (stage, box_width) in enumerate(zip(stages, box_info)):
            # Add the text box at the calculated position
            text = ax.text(current_x + box_width / 2, 0.5, stage, ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='skyblue', alpha=0.5, edgecolor='blue'))
            # Update current position for next box
            current_x += box_width + desired_gap

        # Add arrows between boxes
        current_x = x_start
        for i in range(len(stages) - 1):
            current_width = box_info[i]
            next_width = box_info[i + 1]

            # Calculate arrow start and end positions
            arrow_start = current_x + current_width
            arrow_end = arrow_start + desired_gap

            ax.annotate(
                '',
                xy=(arrow_end, 0.5),
                xytext=(arrow_start, 0.5),
                arrowprops=dict(arrowstyle='->', lw=1)
            )

            current_x += current_width + desired_gap

        # Hide axis
        ax.set_axis_off()
        plt.tight_layout()

        pipeline_canvas = FigureCanvas(pipeline_fig)
        pipeline_canvas.setFixedHeight(75)
        pipeline_canvas.setFixedWidth(1000)        
        pipeline_layout.addWidget(pipeline_canvas)
        
        # return pipeline_widget
        return self.create_scrollable_subwindow("Pipeline Diagram", pipeline_widget)

    def create_results_subwindow(self, results_df, best_model_idx):
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Assuming results_df and best_model_idx are defined globally or within the class
        highlight_style = '<style> .highlight {background-color: yellow;} </style>'
        results_html = results_df.to_html(index=False).replace(
            '<tr>',
            f'<tr class="highlight">' if results_df.index[0] == best_model_idx else '<tr>',
            1
        )

        results_text = QLabel(highlight_style + results_html)
        results_text.setAlignment(Qt.AlignCenter)
        results_text.setStyleSheet("font-size: 10pt;")
        results_layout.addWidget(results_text)

        # return results_widget
        return self.create_scrollable_subwindow("Model Evaluation Results", results_widget)

    def create_importance_subwindow(self, perm_importance_dict):
        importance_widget = QWidget()
        importance_layout = QVBoxLayout(importance_widget)

        for model_name, importance_df in perm_importance_dict.items():
            fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size for better visibility
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
            ax.set_title(f'Permutation Importance for {model_name}')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            plt.tight_layout()

            # Use a FigureCanvas for matplotlib figure
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(400)  # Set a minimum height for better scrolling experience
            canvas.setMinimumWidth(600)  # Set a minimum width for better scrolling experience
            importance_layout.addWidget(canvas)

        return self.create_scrollable_subwindow("Permutation Importance", importance_widget, init_zoom_out_factor=0.7)

    def create_confusion_matrix_subwindow(self, confusion_matrices):
        cm_widget = QWidget()
        cm_layout = QVBoxLayout(cm_widget)

        for model_name, cm in confusion_matrices.items():
            fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size for better visibility
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix for {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()

            # Use a FigureCanvas for matplotlib figure
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(400)  # Set a minimum height for better scrolling experience
            canvas.setMinimumWidth(600)  # Set a minimum width for better scrolling experience
            cm_layout.addWidget(canvas)

        return self.create_scrollable_subwindow("Confusion Matrices", cm_widget, init_zoom_out_factor=0.7)

if __name__ == '__main__':
    import dfhelper

    app = QApplication(sys.argv)
    file = "data/Employee-Attrition.csv"
    df = pd.read_csv(file)
    proj = MlProject()
    proj.loadCSV("data/Employee-Attrition.csv")
    proj.columns = dfhelper.createColumnDict(df)
    proj.targetVariable = 'Attrition'
    df, _ = pre_process(df=df, proj=proj)
    support = training_and_evaluation(df=df, proj=proj)
    support['params'] = {
            'impute': True,
            'remove_invariants': True,
            'handle_outliers': True,
            'vif_threshold': 5,
            'encoding': 'onehot'
        }
    model_eval = ModelEvaluationApp(support)
    model_eval.show()
    sys.exit(app.exec_())
