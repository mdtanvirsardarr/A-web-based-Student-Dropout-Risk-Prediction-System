import matplotlib
matplotlib.use('Agg')  #  server execution
import re
import unicodedata
from flask import (
    Flask, render_template, request, redirect, url_for, flash, session,
    send_from_directory, abort, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func  

import os, io, json, atexit, pickle, warnings, base64, random
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask_cors import CORS
from sqlalchemy import func, or_  
from sqlalchemy import and_  
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
# dependencies 
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler
    IMB_READY = True
except Exception:
    IMB_READY = False

try:
    import shap
    SHAP_READY = True
except Exception:
    SHAP_READY = False
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# App configuration
app = Flask(
    __name__,
    static_folder='.',     
    template_folder='.'     
)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOADS_DIR = os.path.join(basedir, 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.config['ADMIN_PICS'] = UPLOADS_DIR
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MODEL_PATH'] = os.path.join(basedir, 'dropout_model.pkl')
app.config['ARTIFACTS_PATH'] = os.path.join(basedir, 'model_artifacts.json')
app.config['CALIB_PNG'] = os.path.join(basedir, 'calibration_reliability.png')
db = SQLAlchemy(app)
warnings.filterwarnings("ignore", category=UserWarning)

# Database models
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100), nullable=False, default='Administrator')
    department = db.Column(db.String(100), nullable=True, default='Administration')
    profile_photo = db.Column(db.String(200), nullable=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    department = db.Column(db.String(100), nullable=True)
    profile_photo = db.Column(db.String(200), nullable=True)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    attendance = db.Column(db.Float, nullable=False)
    gpa = db.Column(db.Float, nullable=False)
    assignment_avg = db.Column(db.Float, nullable=False)
    quiz_avg = db.Column(db.Float, nullable=False)
    dropout_risk = db.Column(db.String(50), nullable=True)
    last_activity = db.Column(db.DateTime, nullable=True)
    financial_status = db.Column(db.String(50), nullable=True)
    mental_health_score = db.Column(db.Integer, nullable=True)
    extracurricular_activity = db.Column(db.Boolean, nullable=True)

class ModelMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    features_used = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    test_accuracy = db.Column(db.Float)
    test_precision = db.Column(db.Float)
    test_recall = db.Column(db.Float)
    confusion_matrix = db.Column(db.Text)       
    risk_distribution = db.Column(db.Text)      
    feature_importance = db.Column(db.Text)     

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='CASCADE'), 
        nullable=False
    )
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    prediction = db.Column(db.String(16), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    features = db.Column(db.Text, nullable=False)
    probabilities = db.Column(db.Text, nullable=False)
    user = db.relationship(
        'User',
        backref=db.backref(
            'predictions',
            lazy=True,
            cascade='all, delete-orphan',   
            passive_deletes=True           
        )
    )
    
class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Staff(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    department_id = db.Column(db.Integer,
                              db.ForeignKey('department.id', ondelete='CASCADE'),
                              nullable=False)
    department = db.relationship(
        'Department',
        backref=db.backref('staff', lazy=True, cascade='all, delete-orphan', passive_deletes=True)
    )
    profile_photo = db.Column(db.String(200), nullable=True)
    
class EmailLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    from_role = db.Column(db.String(16), nullable=False)             
    from_email = db.Column(db.String(120), nullable=False)
    to_email   = db.Column(db.String(120), nullable=False)
    subject    = db.Column(db.String(200))
    body       = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    read_by_recipient  = db.Column(db.Boolean, default=False)     
    deleted_by_sender  = db.Column(db.Boolean, default=False)     
    deleted_by_recipient = db.Column(db.Boolean, default=False)  
    
def _current_sender():
    if session.get("admin_logged_in"):
        return "admin", (session.get("admin_email") or "").strip().lower()
    if session.get("staff_logged_in"):
        return "staff", (session.get("staff_email") or "").strip().lower()
    if session.get("user_logged_in") or session.get("student_logged_in"):
        return "user", (session.get("user_email") or session.get("student_email") or "").strip().lower()
    return None, None

def _whoami_email():
    role, email = _current_sender()
    if not role or not email:
        abort(401)
    return role, email

def _whoami_from_request():
    prefer = (request.args.get("as") or "").strip().lower()
    if prefer == "staff" and session.get("staff_logged_in"):
        return "staff", (session.get("staff_email") or "").strip().lower()
    if prefer == "admin" and session.get("admin_logged_in"):
        return "admin", (session.get("admin_email") or "").strip().lower()
    if prefer == "user" and session.get("user_logged_in"):
        return "user", (session.get("user_email") or "").strip().lower()
    return _whoami_email()

def _lookup_contact(email: str):
    e = (email or "").strip().lower()
    if not e:
        return {"email": email}
    s = Staff.query.filter(func.lower(Staff.email) == e).first()
    if s:
        return {
            "email": s.email, "name": s.name, "role": "staff",
            "department": (s.department.slug if s.department else None)
        }
    a = Admin.query.filter(func.lower(Admin.email) == e).first()
    if a:
        return {"email": a.email, "name": a.name or "Admin", "role": "admin", "department": "administration"}
    u = User.query.filter(func.lower(User.email) == e).first()
    if u:
        return {"email": u.email, "name": u.name, "role": "user", "department": u.department}
    return {"email": email}
    
def _email_to_json(row: EmailLog, me: str):
    me = (me or "").lower()
    is_sender = (row.from_email or "").lower() == me
    is_rcpt   = (row.to_email or "").lower() == me
    box = "sent" if is_sender and not row.deleted_by_sender else "inbox" if is_rcpt and not row.deleted_by_recipient else "trash"
    return {
        "id": row.id,
        "subject": row.subject or "",
        "body": row.body or "",
        "created_at": row.created_at.isoformat(),
        "from": _lookup_contact(row.from_email),
        "to": _lookup_contact(row.to_email),
        "box": box,
        "read": (True if is_sender else bool(row.read_by_recipient)), 
        "deleted_by_sender": bool(row.deleted_by_sender),
        "deleted_by_recipient": bool(row.deleted_by_recipient),
    }

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass

def _slugify(value: str):
    if not value:
        return ''
    s = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')
    s = s.strip().lower()
    s = re.sub(r'[\s/]+', '_', s)
    s = re.sub(r'[^a-z0-9_-]', '', s)
    s = re.sub(r'[_-]{2,}', '_', s)
    return s[:64]

def get_factor_status(name: str, value: float):
    if name == 'attendance':
        return 'Low' if value >= 80 else 'Medium' if value >= 70 else 'High'
    if name == 'gpa':
        return 'Low' if value >= 3.0 else 'Medium' if value >= 2.5 else 'High'
    if name == 'assignment_avg':
        return 'Low' if value >= 70 else 'Medium' if value >= 60 else 'High'
    if name == 'quiz_avg':
        return 'Low' if value >= 70 else 'Medium' if value >= 60 else 'High'
    if name == 'mental_health_score':
        return 'Low' if value >= 7 else 'Medium' if value >= 5 else 'High'
    if name == 'days_since_last_activity':
        return 'Low' if value <= 7 else 'Medium' if value <= 30 else 'High'
    if name == 'extracurricular_activity':
        return 'Low' if bool(value) else 'Medium'
    return 'Medium'

def ensure_utc(dt):
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

# Synthetic data generation
fake = None
def _fake():
    global fake
    if fake is None:
        try:
            from faker import Faker
            fake = Faker()
        except Exception:
            class _Mini:
                def name(self): return f"Student {random.randint(1,99999)}"
                def email(self): return f"s{random.randint(1,99999)}@example.com"
            fake = _Mini()
    return fake

def generate_mental_health():
    return int(np.clip(np.round(np.random.normal(6.8, 1.8)), 1, 10))

def financial_status_from_gpa(gpa):
    return 'Stable' if gpa >= 2.7 else ('Scholarship' if gpa >= 3.5 else 'Unstable')

def generate_extracurricular(gpa):
    return bool(np.random.rand() < (0.75 if gpa >= 3.0 else 0.45))

def generate_random_students(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        name = _fake().name()
        email = _fake().email()
        attendance = float(np.clip(rng.normal(84, 10), 40, 100))
        gpa = float(np.clip(rng.normal(3.0, 0.5), 0.5, 4.0))
        assignment_avg = float(np.clip(rng.normal(70, 12), 0, 100))
        quiz_avg = float(np.clip(rng.normal(72, 15), 0, 100))
        last_activity = datetime.now() - timedelta(days=int(np.clip(abs(rng.normal(8, 10)), 0, 90)))
        fin = financial_status_from_gpa(gpa)
        mh = generate_mental_health()
        ec = generate_extracurricular(gpa)
        try:
            db.session.add(Student(
                name=name, email=email,
                attendance=attendance, gpa=gpa,
                assignment_avg=assignment_avg, quiz_avg=quiz_avg,
                last_activity=last_activity, financial_status=fin,
                mental_health_score=mh, extracurricular_activity=ec
            ))
            db.session.commit()
        except Exception:
            db.session.rollback()

def _compute_risk_scores(df: pd.DataFrame):
    days = df['days_since_last_activity'].values
    scores = (
        0.30*(100 - df['attendance'].values) +
        0.30*(4.0 - df['gpa'].values)*25 +
        0.10*(100 - df['assignment_avg'].values) +
        0.10*(100 - df['quiz_avg'].values) +
        0.10*np.clip((days - 7)/7, 0, 1) * 100 +
        0.07*df['financial_unstable'].values*100 +
        0.03*(10 - df['mental_health_score'].values)*10
    )
    return scores + np.random.default_rng(42).normal(0, 4, size=len(df))

def _label_from_scores(scores: np.ndarray, target_dist=(0.70, 0.25, 0.05)):
    low_p, med_p, high_p = target_dist
    q_high = 1.0 - high_p
    q_med = 1.0 - (high_p + med_p)
    th_high = np.quantile(scores, q_high)
    th_med = np.quantile(scores, q_med)
    return np.where(scores >= th_high, 'High', np.where(scores >= th_med, 'Medium', 'Low'))

def _binarize(y, classes):
    return label_binarize(y, classes=classes)

def _ece_multiclass(y_true, proba, n_bins=10):
    y_true = np.asarray(y_true)
    cls = np.unique(y_true)
    y_bin = _binarize(y_true, classes=cls)
    eces = []
    for j in range(len(cls)):
        p = proba[:, j]
        t = y_bin[:, j]
        bins = np.linspace(0, 1, n_bins+1)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b+1]
            m = (p >= lo) & (p < hi) if b < n_bins-1 else (p >= lo) & (p <= hi)
            if not np.any(m):
                continue
            conf = p[m].mean()
            acc = t[m].mean()
            ece += np.abs(acc - conf) * (m.mean())
        eces.append(ece)
    return float(np.mean(eces)) if eces else 0.0

def _auprc_macro(y_true, proba, classes):
    Y = label_binarize(y_true, classes=classes)
    return float(average_precision_score(Y, proba, average='macro'))

def _auroc_macro(y_true, proba, classes):
    Y = label_binarize(y_true, classes=classes)
    try:
        return float(roc_auc_score(Y, proba, average='macro', multi_class='ovr'))
    except Exception:
        return float('nan')

def _reliability_plot(y_true, proba, path_png):
    try:
        y_true = np.asarray(y_true)
        classes = np.unique(y_true)
        Y = label_binarize(y_true, classes=classes)
        bins = np.linspace(0,1,11)
        plt.figure(figsize=(6,5))
        for j in range(len(classes)):
            p = proba[:, j]; t = Y[:, j]
            xs, ys = [], []
            for b in range(10):
                lo, hi = bins[b], bins[b+1]
                m = (p >= lo) & (p < hi) if b < 9 else (p >= lo) & (p <= hi)
                if np.any(m):
                    xs.append(p[m].mean()); ys.append(t[m].mean())
            plt.plot(xs, ys, marker='o', label=f'class={classes[j]}')
        plt.plot([0,1],[0,1],'--', linewidth=1)
        plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
        plt.title('Reliability (calibration) diagram')
        plt.legend(); plt.tight_layout()
        plt.savefig(path_png, dpi=160); plt.close()
    except Exception:
        pass

def _threshold_table(y_true, proba, classes, positive_label, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    idx = list(classes).index(positive_label)
    p = proba[:, idx]
    y = (np.asarray(y_true) == positive_label).astype(int)
    rows = []
    for thr in grid:
        pred = (p >= thr).astype(int)
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        rows.append([float(thr), float(prec), float(rec), float(f1)])
    return pd.DataFrame(rows, columns=['threshold','precision','recall','f1'])

def _pick_thresholds_from_val(y_val, proba_val, classes):
    tbl_high = _threshold_table(y_val, proba_val, classes, 'High')
    tbl_low  = _threshold_table(y_val, proba_val, classes, 'Low')

    th_high = float(tbl_high.loc[tbl_high['f1'].idxmax(), 'threshold'])
    th_low  = float(tbl_low.loc[tbl_low['f1'].idxmax(),  'threshold'])

    beta = 0.5
    f05 = (1 + beta**2) * (tbl_low['precision'] * tbl_low['recall']) / (beta**2 * tbl_low['precision'] + tbl_low['recall'] + 1e-12)
    th_low_f05 = float(tbl_low.loc[f05.idxmax(), 'threshold']) if len(tbl_low) > 0 else th_low

    th_high_recall80 = float(tbl_high.loc[(tbl_high['recall'] - 0.80).abs().idxmin(), 'threshold'])

    return {
        "high_f1": th_high, "low_f1": th_low,
        "alt": {"high_target_recall_0.80": th_high_recall80, "low_f0.5": th_low_f05},
        "tables": {"high": tbl_high.to_dict(orient='records'),
                   "low":  tbl_low.to_dict(orient='records')}
    }

def _apply_locked_bands(proba_row: Dict[str,float], thr_high: float, thr_low: float):
    if proba_row.get('High',0.0) >= thr_high:
        return 'High'
    if proba_row.get('Low',0.0) >= thr_low:
        return 'Low'
    return 'Medium'

def _extract_feature_importances_from_classifier(clf):
    imp = getattr(clf, 'feature_importances_', None)
    if imp is None:
        return None
    return np.asarray(imp, dtype=float)

# Core training with leakage-free nested cross-validation
def _build_base_pipeline():
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    if IMB_READY:
        steps.append(('sampler', RandomOverSampler(random_state=42)))
        pipe_cls = ImbPipeline
    else:
        pipe_cls = Pipeline
    steps.append(('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')))
    return pipe_cls(steps)

def _param_space():
    return {
        'classifier__n_estimators': [300, 500, 800, 1200],
        'classifier__max_depth': [8, 12, 16, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
    }

def _tune_inner(X, y, n_splits=3, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        _build_base_pipeline(),
        param_distributions=_param_space(),
        n_iter=25, scoring='f1_weighted', cv=cv, n_jobs=-1, refit=True, random_state=random_state, verbose=0
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

def _choose_calibrator(estimator, X_train, y_train, X_val, y_val):
    methods = ['isotonic', 'sigmoid']
    results = []
    for m in methods:
        pipe_cls = ImbPipeline if IMB_READY else Pipeline
        cal = pipe_cls(list(estimator.steps[:-1]) + [
            ('classifier', CalibratedClassifierCV(
                estimator=estimator.named_steps['classifier'], method=m, cv=3
            ))
        ])
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_val)
        classes = cal.named_steps['classifier'].classes_
        ece = _ece_multiclass(y_val, proba, n_bins=10)
        auprc = _auprc_macro(y_val, proba, classes)
        results.append((m, ece, auprc, cal))
    results.sort(key=lambda r: (r[1], -r[2]))  
    return results[0]  

def train_dropout_model():
    with app.app_context():
        students = Student.query.all()
        if len(students) < 60:
            raise RuntimeError("Not enough students to train a model.")

        data = {
            'attendance': [s.attendance for s in students],
            'gpa': [s.gpa for s in students],
            'assignment_avg': [s.assignment_avg for s in students],
            'quiz_avg': [s.quiz_avg for s in students],
            'days_since_last_activity': [(datetime.now() - (s.last_activity or datetime.min)).days for s in students],
            'financial_stable': [1 if s.financial_status == 'Stable' else 0 for s in students],
            'financial_unstable': [1 if s.financial_status == 'Unstable' else 0 for s in students],
            'mental_health_score': [s.mental_health_score or 5 for s in students],
            'extracurricular_activity': [1 if s.extracurricular_activity else 0 for s in students]
        }
        df = pd.DataFrame(data)

        risk_scores = _compute_risk_scores(df)
        labels = _label_from_scores(risk_scores, target_dist=(0.70, 0.25, 0.05))
        df['dropout_risk'] = labels

        X = df.drop(columns=['dropout_risk']).astype(float)
        y = df['dropout_risk'].values
        classes = np.unique(y)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_metrics = []
        all_val_probs, all_val_true = [], []

        for outer_train_idx, outer_val_idx in outer.split(X_trainval, y_trainval):
            X_tr, X_val = X_trainval.iloc[outer_train_idx], X_trainval.iloc[outer_val_idx]
            y_tr, y_val = y_trainval[outer_train_idx], y_trainval[outer_val_idx]

            best_estimator, _ = _tune_inner(X_tr, y_tr, n_splits=3, random_state=42)

            method, ece_val, auprc_val, cal_pipeline = _choose_calibrator(best_estimator, X_tr, y_tr, X_val, y_val)

            y_pred = cal_pipeline.predict(X_val)
            proba = cal_pipeline.predict_proba(X_val)

            macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)
            prec_w = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec_w = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            auroc = _auroc_macro(y_val, proba, classes)
            auprc = _auprc_macro(y_val, proba, classes)

            Ybin = label_binarize(y_val, classes=classes)
            briers = []
            for j in range(len(classes)):
                briers.append(brier_score_loss(Ybin[:, j], proba[:, j]))
            brier = float(np.mean(briers))
            ece = _ece_multiclass(y_val, proba)

            outer_metrics.append({
                "accuracy": acc, "precision_w": prec_w, "recall_w": rec_w,
                "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
                "auroc_macro": auroc, "auprc_macro": auprc,
                "brier": brier, "ece": ece, "cal_method": method
            })
            all_val_probs.append(proba); all_val_true.append(y_val)

        def agg(key):
            arr = np.array([m[key] for m in outer_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))])
            if arr.size:
                return (float(arr.mean()), float(arr.std()))
            else:
                return (float('nan'), float('nan'))

        nested_summary = {k: {"mean": agg(k)[0], "std": agg(k)[1]} for k in outer_metrics[0].keys() if k != "cal_method"}
        nested_summary["cal_method_majority"] = max(
            set([m["cal_method"] for m in outer_metrics]),
            key=[m["cal_method"] for m in outer_metrics].count
        )

        all_val_probs = np.vstack(all_val_probs)
        all_val_true = np.concatenate(all_val_true)
        thresholds = _pick_thresholds_from_val(all_val_true, all_val_probs, classes)

        X_tr, X_val, y_tr, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=7)
        final_estimator, final_params = _tune_inner(X_tr, y_tr, n_splits=3, random_state=123)
        method, ece_val, auprc_val, final_pipeline = _choose_calibrator(final_estimator, X_tr, y_tr, X_val, y_val)

        pipe_cls = ImbPipeline if IMB_READY else Pipeline
        final_pipeline = pipe_cls(list(final_estimator.steps[:-1]) + [
            ('classifier', CalibratedClassifierCV(
                estimator=final_estimator.named_steps['classifier'],
                method=method, cv=5
            ))
        ])
        final_pipeline.fit(X_trainval, y_trainval)

        y_pred_test = final_pipeline.predict(X_test)
        proba_test = final_pipeline.predict_proba(X_test)

        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_rec  = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_macro_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_bal_acc  = balanced_accuracy_score(y_test, y_pred_test)
        test_auprc    = _auprc_macro(y_test, proba_test, classes)
        test_auroc    = _auroc_macro(y_test, proba_test, classes)

        Ybin_t = label_binarize(y_test, classes=classes)
        briers = [brier_score_loss(Ybin_t[:, j], proba_test[:, j]) for j in range(len(classes))]
        test_brier = float(np.mean(briers))
        test_ece   = _ece_multiclass(y_test, proba_test)

        _reliability_plot(y_test, proba_test, app.config['CALIB_PNG'])

        tuned_rf = final_estimator.named_steps['classifier']
        importances = _extract_feature_importances_from_classifier(tuned_rf)
        feat_names = list(X.columns)
        if importances is None:
            try:
                r = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
                importances = r.importances_mean
            except Exception:
                importances = np.zeros(len(feat_names))
        feature_imp = dict(zip(feat_names, [float(x) for x in importances]))

        shap_summary = {}
        if SHAP_READY:
            try:
                explainer = shap.TreeExplainer(tuned_rf)
                sv = explainer.shap_values(X_trainval)
                if isinstance(sv, list):
                    mean_abs = np.mean(np.abs(np.array(sv)), axis=(0,1))
                else:
                    mean_abs = np.mean(np.abs(sv), axis=0)
                shap_summary = dict(zip(feat_names, [float(x) for x in mean_abs]))
            except Exception:
                pass

        with open(app.config['MODEL_PATH'], 'wb') as f:
            pickle.dump(final_pipeline, f)

        preds_all = final_pipeline.predict_proba(X)
        classes_list = list(final_pipeline.named_steps['classifier'].classes_)
        idxH = classes_list.index('High'); idxL = classes_list.index('Low'); idxM = classes_list.index('Medium')
        thrH = thresholds["alt"]["high_target_recall_0.80"]
        thrL = thresholds["low_f1"]
        for stu, prows in zip(students, preds_all):
            prob_row = {'High': float(prows[idxH]), 'Medium': float(prows[idxM]), 'Low': float(prows[idxL])}
            stu.dropout_risk = _apply_locked_bands(prob_row, thrH, thrL)
        db.session.commit()

        mm = ModelMetrics(
            accuracy=test_acc, precision=test_prec, recall=test_rec,
            features_used=", ".join(feat_names),
            test_accuracy=test_acc, test_precision=test_prec, test_recall=test_rec,
            confusion_matrix=json.dumps(confusion_matrix(y_test, y_pred_test).tolist()),
            risk_distribution=json.dumps({
                "Low": int(np.sum(y_test=='Low')),
                "Medium": int(np.sum(y_test=='Medium')),
                "High": int(np.sum(y_test=='High'))
            }),
            feature_importance=json.dumps(feature_imp)
        )
        db.session.add(mm); db.session.commit()

        artifacts = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "RandomForestClassifier (calibrated)",
            "oversampling": "RandomOverSampler in CV" if IMB_READY else "Not available (install imblearn)",
            "calibration": {"chosen_method": method, "val_ece": ece_val, "val_auprc": auprc_val},
            "nested_cv": nested_summary,
            "locked_thresholds": thresholds,
            "test_metrics": {
                "accuracy": test_acc, "precision_w": test_prec, "recall_w": test_rec,
                "macro_f1": test_macro_f1, "balanced_accuracy": test_bal_acc,
                "auroc_macro": test_auroc, "auprc_macro": test_auprc,
                "brier": test_brier, "ece": test_ece
            },
            "feature_importance": feature_imp,
            "shap_summary": shap_summary,
        }
        with open(app.config['ARTIFACTS_PATH'], 'w') as f:
            json.dump(artifacts, f, indent=2)

        print(f"Model trained. Test macro-F1={test_macro_f1:.3f}, AUROC={test_auroc:.3f}, AUPRC={test_auprc:.3f} (method={method})", flush=True)
        return artifacts

# User prediction and save history
@app.route('/api/user/predict', methods=['POST'])
def api_user_predict():
    user = _current_user_or_401()
    if not user:
        return jsonify({"error":"Not authenticated"}), 401

    latest = ModelMetrics.query.order_by(ModelMetrics.created_at.desc()).first()
    if not latest or not os.path.exists(app.config['MODEL_PATH']): 
        return jsonify({"error":"Model not trained yet"}), 503

    with open(app.config['MODEL_PATH'], 'rb') as f:
        model = pickle.load(f)

    payload = request.get_json(force=True)

    features = {
        'attendance': float(payload['attendance']),
        'gpa': float(payload['gpa']),
        'assignment_avg': float(payload['assignment_avg']),
        'quiz_avg': float(payload['quiz_avg']),
        'days_since_last_activity': float(payload['days_since_last_activity']),
        'financial_status': payload.get('financial_status', 'Stable'),
        'financial_stable': 1 if payload.get('financial_status') == 'Stable' else 0,
        'financial_unstable': 1 if payload.get('financial_status') == 'Unstable' else 0,
        'mental_health_score': int(payload.get('mental_health_score', 6)),
        'extracurricular_activity': 1 if payload.get('extracurricular_activity', False) else 0
    }

    X = pd.DataFrame([{
        'attendance': features['attendance'],
        'gpa': features['gpa'],
        'assignment_avg': features['assignment_avg'],
        'quiz_avg': features['quiz_avg'],
        'days_since_last_activity': features['days_since_last_activity'],
        'financial_stable': features['financial_stable'],
        'financial_unstable': features['financial_unstable'],
        'mental_health_score': features['mental_health_score'],
        'extracurricular_activity': features['extracurricular_activity']
    }])

    proba = model.predict_proba(X)[0]
    labels = list(model.named_steps['classifier'].classes_)
    prob_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

    thr_high, thr_low = 0.6, 0.6
    if os.path.exists(app.config['ARTIFACTS_PATH']):
        with open(app.config['ARTIFACTS_PATH'], 'r') as f:
            art = json.load(f)
        locked = art.get('locked_thresholds', {})
        thr_high = locked.get('alt', {}).get('high_target_recall_0.80',
                    locked.get('high_f1', 0.6))
        thr_low = locked.get('low_f1', 0.6)

    band = _apply_locked_bands(prob_dict, thr_high, thr_low)
    risk_score = round(prob_dict.get('High',0)*100 + prob_dict.get('Medium',0)*50, 2)

    row = PredictionHistory(
        user_id=user.id,
        prediction=band,
        risk_score=risk_score,
        features=json.dumps(features),
        probabilities=json.dumps(prob_dict)
    )
    db.session.add(row); db.session.commit()

    artifacts = {}
    if os.path.exists(app.config['ARTIFACTS_PATH']):
        with open(app.config['ARTIFACTS_PATH'],'r') as f:
            artifacts = json.load(f)

    # SHAP explanation
    shap_values_dict = None
    if SHAP_READY:
        try:
            clf = model.named_steps.get('classifier')
            rf = None
            # CalibratedClassifierCV keeps per-fold calibrated estimators
            if hasattr(clf, 'calibrated_classifiers_') and clf.calibrated_classifiers_:
                cc = clf.calibrated_classifiers_[0]
                rf = getattr(cc, 'base_estimator', None) or getattr(cc, 'estimator', None)
            # Fallback attribute names 
            if rf is None:
                rf = getattr(clf, 'base_estimator', None) or getattr(clf, 'estimator', None)
            if rf is not None:
                explainer = shap.TreeExplainer(rf)
                sv = explainer.shap_values(X)
                # choose class index matching band 
                class_labels = list(getattr(rf, "classes_", [])) or labels
                try:
                    sel_idx = class_labels.index(band)
                except ValueError:
                    sel_idx = int(np.argmax(proba))
                if isinstance(sv, list):
                    vec = sv[sel_idx][0]
                else:
                    vec = sv[0]
                shap_values_dict = {col: float(vec[i]) for i, col in enumerate(X.columns)}
        except Exception as e:
            print(f"SHAP per-instance failed: {e}", flush=True)
            shap_values_dict = None
    return jsonify({
        "prediction": band,
        "probabilities": prob_dict,
        "risk_score": risk_score,
        "locked_thresholds": {"high": thr_high, "low": thr_low},
        "model_meta": {
            "accuracy": latest.accuracy, "precision": latest.precision,
            "recall": latest.recall, "last_trained": latest.created_at.isoformat()
        },
        "calibration": artifacts.get("calibration"),
        "model": artifacts.get("model") or "RandomForestClassifier (calibrated)",
        "shap_values": shap_values_dict,                     
        "explanations": {"shap_values": shap_values_dict}    
    })

@app.route('/api/user/predictions')
def api_user_predictions():
    user = _current_user_or_401()
    if not user:
        return jsonify({"error":"Not authenticated"}), 401

    limit = int(request.args.get('limit', 100))
    q = (PredictionHistory.query
         .filter_by(user_id=user.id)
         .order_by(PredictionHistory.created_at.desc())
         .limit(limit).all())

    items = []
    for r in q:
        # Normalize legacy 0–1 scores to 0–100
        raw = r.risk_score if r.risk_score is not None else 0.0
        score = raw * 100.0 if raw <= 1.0 else raw
        items.append({
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "prediction": r.prediction,
            "risk_score": round(score, 2),
            "features": json.loads(r.features),
            "probabilities": json.loads(r.probabilities)
        })

    artifacts = {}
    if os.path.exists(app.config['ARTIFACTS_PATH']):
        with open(app.config['ARTIFACTS_PATH'],'r') as f:
            artifacts = json.load(f)

    return jsonify({
        "items": items,
        "summary": {"count": PredictionHistory.query.filter_by(user_id=user.id).count()},
        "artifacts": artifacts
    })
    
# Delete prediction the user owns 
@app.route('/api/user/predictions/<int:pred_id>', methods=['DELETE'])
def api_user_prediction_delete(pred_id):
    user = _current_user_or_401()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    row = PredictionHistory.query.filter_by(id=pred_id, user_id=user.id).first()
    if not row:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(row)
    db.session.commit()
    return jsonify({"ok": True})

# clear ALL predictions for current user 
@app.route('/api/user/predictions/clear', methods=['DELETE'])
def api_user_predictions_clear():
    user = _current_user_or_401()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    PredictionHistory.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return jsonify({"ok": True})

# Authentication 
def _current_user_or_401():
    if not session.get('user_logged_in'):
        return None
    email = (session.get('user_email') or '').strip().lower()
    if not email:
        return None
    return User.query.filter(func.lower(User.email) == email).first()

# Views of public pages
@app.route('/')
def root_index(): return render_template('index.html')

@app.route('/index.html')
def serve_index_html(): return render_template('index.html')

@app.route('/about.html')
def serve_about_html(): return render_template('about.html')

# User authentication
@app.route('/auth.html')
def auth_page(): return render_template('auth.html')

@app.route('/user/register', methods=['POST'], strict_slashes=False)
def user_register():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        name = (data.get('name') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
    else:
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
    if not all([name, email, password]):
        if request.is_json:
            return jsonify({"ok": False, "error": "All fields required"}), 400
        flash('All fields required', 'danger'); return redirect(url_for('auth_page'))
    if User.query.filter(func.lower(User.email) == email).first():
        if request.is_json:
            return jsonify({"ok": False, "error": "Email already registered"}), 409
        flash('Email already registered', 'danger'); return redirect(url_for('auth_page'))
    u = User(name=name, email=email, password_hash=generate_password_hash(password))
    db.session.add(u); db.session.commit()
    if request.is_json:
        return jsonify({"ok": True, "user_id": u.id}), 201
    flash('Registration successful!', 'success'); return redirect(url_for('auth_page'))

# login page accepts both /user/login and /login (POST)
@app.route('/user/login', methods=['POST'])
@app.route('/login', methods=['POST'])
def user_login():
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    u = User.query.filter(func.lower(User.email) == email).first()
    if u and check_password_hash(u.password_hash, password):
        session['user_logged_in'] = True; session['user_email'] = u.email
        return redirect(url_for('user_profile'))
    flash('Invalid email or password', 'danger'); return redirect(url_for('auth_page'))

@app.route('/user/logout')
def user_logout():
    session.pop('user_logged_in', None); session.pop('user_email', None)
    return redirect(url_for('auth_page'))

# User profile (protected)
@app.route('/user-profile')
def user_profile():
    if not session.get('user_logged_in'):
        return redirect(url_for('auth_page'))
    user = User.query.filter_by(email=session['user_email']).first()
    return render_template('user-profile.html', current_user=user)

@app.route('/user-profile.html')
def user_profile_html():
    return redirect(url_for('user_profile'))

@app.route('/user-profile/update', methods=['POST'])
def update_user_profile():
    if not session.get('user_logged_in'):
        return redirect(url_for('auth_page'))
    user = User.query.filter_by(email=session['user_email']).first()
    user.name = request.form.get('name', user.name)
    user.department = request.form.get('department', user.department)
    if 'photo' in request.files:
        f = request.files['photo']
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['ADMIN_PICS'], filename)
            f.save(path)
            user.profile_photo = f'/uploads/{filename}'
    db.session.commit()
    flash('Profile updated!', 'success')
    return redirect(url_for('user_profile'))

# Admin authentication and dashboard
@app.route('/admin-login.html')
def serve_admin_login_html(): return render_template('admin-login.html')

# Convenience aliases for registration routes
@app.route('/register', methods=['GET', 'POST'], strict_slashes=False)
def register_alias():
    if request.method == 'GET':
        return redirect(url_for('auth_page'))
    return user_register()

@app.route('/register.html')
def register_page_alias():
    return redirect(url_for('auth_page'))

# Accepts both /admin/login and /admin-login (POST)
@app.route('/admin/login', methods=['POST'])
@app.route('/admin-login', methods=['POST'])
def admin_login():
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    admin = Admin.query.filter(func.lower(Admin.email) == email).first()
    if admin and check_password_hash(admin.password_hash, password):
        session['admin_logged_in'] = True; session['admin_email'] = (admin.email or '').strip().lower()
        return redirect(url_for('admin_dashboard'))
    flash('Invalid admin email or password', 'danger'); return redirect(url_for('serve_admin_login_html'))

def _require_admin():
    if not session.get('admin_logged_in'):
        abort(403)

@app.route('/admin-dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('serve_admin_login_html'))
    admin = Admin.query.filter(func.lower(Admin.email) == (session.get('admin_email') or '').strip().lower()).first()
    return render_template('admin-dashboard.html', current_user=admin)

@app.route('/admin-dashboard.html')
def admin_dashboard_html():
    return redirect(url_for('admin_dashboard'))

# Admin profile (GET/PUT)
@app.route('/api/admin/me', methods=['GET', 'PUT'])
def api_admin_me():
    _require_admin()
    admin = Admin.query.filter(func.lower(Admin.email) == (session.get('admin_email') or '').strip().lower()).first_or_404()
    if request.method == 'GET':
        return jsonify({
            "id": admin.id,
            "name": admin.name,
            "email": admin.email,
            "department": admin.department,
            "profile_photo": admin.profile_photo or '/uploads/default.png'
        })

    if request.content_type and 'multipart/form-data' in request.content_type:
        form = request.form
        admin.name = form.get('name', admin.name)
        new_email = (form.get('email') or admin.email).strip().lower()
        admin.department = form.get('department', admin.department)

        if 'profile_photo' in request.files:
            f = request.files['profile_photo']
            if f and f.filename:
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['ADMIN_PICS'], filename)
                f.save(path)
                admin.profile_photo = f'/uploads/{filename}'

        email_changed = (new_email != admin.email)
        admin.email = new_email
        db.session.commit()
        if email_changed:
            session['admin_email'] = admin.email
        return jsonify({"ok": True})

    data = request.get_json(force=True) or {}
    admin.name = data.get('name', admin.name)
    new_email = (data.get('email') or admin.email).strip().lower()
    admin.department = data.get('department', admin.department)
    email_changed = (new_email != admin.email)
    admin.email = new_email
    db.session.commit()
    if email_changed:
        session['admin_email'] = admin.email
    return jsonify({"ok": True})

# Admin users list
@app.route('/api/admin/users', methods=['GET'])
def api_admin_users():
    _require_admin()
    users = User.query.order_by(User.id.desc()).all()
    out = []
    for u in users:
        last = (PredictionHistory.query
                .filter_by(user_id=u.id)
                .order_by(PredictionHistory.created_at.desc())
                .first())
        if last:
            raw = last.risk_score or 0.0
            risk100 = raw * 100.0 if raw <= 1.0 else raw  
            last_band = last.prediction
            last_risk = round(float(risk100), 1)          
            last_time = last.created_at.isoformat()
        else:
            last_band = None
            last_risk = None
            last_time = None

        out.append({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "department": u.department,
            "profile_photo": u.profile_photo,
            "last_band": last_band,       
            "last_risk": last_risk,      
            "last_time": last_time        
        })
    return jsonify(out)


# Admin user modification API
@app.route('/api/admin/users/<int:user_id>', methods=['PUT','DELETE'])
def api_admin_user_modify(user_id):
    _require_admin()
    u = User.query.get_or_404(user_id)

    if request.method == 'DELETE':
        db.session.delete(u)
        db.session.commit()
        return jsonify({"ok": True})

    if request.content_type and 'multipart/form-data' in request.content_type:
        form = request.form
        u.name  = form.get('name', u.name)
        new_email = (form.get('email') or u.email).strip().lower()

        if 'photo' in request.files:
            f = request.files['photo']
            if f and f.filename:
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['ADMIN_PICS'], filename)
                f.save(path)
                u.profile_photo = f'/uploads/{filename}'

        u.email = new_email
        u.department = form.get('department', u.department)
        db.session.commit()
        return jsonify({"ok": True})

    data = request.get_json(force=True) or {}
    u.name = data.get('name', u.name)
    u.email = (data.get('email') or u.email).strip().lower()
    u.department = data.get('department', u.department)
    db.session.commit()
    return jsonify({"ok": True})

# Students API
@app.route('/api/students')
def api_students():
    students = Student.query.order_by(Student.id.desc()).all()
    return jsonify([{
        "id": s.id, "name": s.name, "email": s.email,
        "attendance": s.attendance, "gpa": s.gpa,
        "assignment_avg": s.assignment_avg, "quiz_avg": s.quiz_avg,
        "dropout_risk": s.dropout_risk,
        "last_activity_days": (datetime.now() - (s.last_activity or datetime.min)).days,
        "financial_status": s.financial_status,
        "mental_health_score": s.mental_health_score,
        "extracurricular_activity": bool(s.extracurricular_activity)
    } for s in students])

# Dashboard model analysis
@app.route('/api/dropout-analysis')
def dropout_analysis():
    try:
        latest = ModelMetrics.query.order_by(ModelMetrics.created_at.desc()).first()
        if not latest: return jsonify({"error":"No model metrics found"}), 404
        if not os.path.exists(app.config['MODEL_PATH']): return jsonify({"error":"Model not found"}), 404

        with open(app.config['MODEL_PATH'], 'rb') as f:
            pipeline = pickle.load(f)

        students = Student.query.all()
        risk_counts = {
            'High': sum(1 for s in students if s.dropout_risk == 'High'),
            'Medium': sum(1 for s in students if s.dropout_risk == 'Medium'),
            'Low': sum(1 for s in students if s.dropout_risk == 'Low')
        }

        try:
            feature_importance = json.loads(latest.feature_importance)
        except Exception:
            clf = getattr(pipeline, 'named_steps', {}).get('classifier', None)
            imp = _extract_feature_importances_from_classifier(clf) or []
            feature_importance = {f'feat_{i}': float(v) for i, v in enumerate(imp)}

        artifacts = {}
        if os.path.exists(app.config['ARTIFACTS_PATH']):
            with open(app.config['ARTIFACTS_PATH'],'r') as f:
                artifacts = json.load(f)

        return jsonify({
            "risk_counts": risk_counts,
            "metrics": {
                "accuracy": latest.accuracy,
                "precision": latest.precision,
                "recall": latest.recall,
                "last_trained": latest.created_at.isoformat()
            },
            "confusion_matrix": json.loads(latest.confusion_matrix or "[]"),
            "feature_importance": feature_importance,
            "artifacts": artifacts,
            "calibration_plot": os.path.basename(app.config['CALIB_PNG']) if os.path.exists(app.config['CALIB_PNG']) else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Per-student factor breakdown
@app.route('/api/student-risk-factors/<int:student_id>')
def student_risk_factors(student_id):
    s = Student.query.get_or_404(student_id)
    factors = {
        'attendance': s.attendance,
        'gpa': s.gpa,
        'assignment_avg': s.assignment_avg,
        'quiz_avg': s.quiz_avg,
        'days_since_last_activity': (datetime.now() - (s.last_activity or datetime.min)).days,
        'financial_stable': 1 if s.financial_status == 'Stable' else 0,
        'financial_unstable': 1 if s.financial_status == 'Unstable' else 0,
        'mental_health_score': s.mental_health_score or 5,
        'extracurricular_activity': 1 if s.extracurricular_activity else 0
    }
    legend = {
        'attendance': {'value': factors['attendance'],
                       'status': get_factor_status('attendance', factors['attendance']),
                       'thresholds': {'High':'<70','Medium':'70–79','Low':'≥80'}},
        'gpa': {'value': factors['gpa'],
                'status': get_factor_status('gpa', factors['gpa']),
                'thresholds': {'High':'<2.5','Medium':'2.5–2.9','Low':'≥3.0'}},
        'assignment': {'value': factors['assignment_avg'],
                       'status': get_factor_status('assignment_avg', factors['assignment_avg']),
                       'thresholds': {'High':'<60','Medium':'60–69','Low':'≥70'}},
        'quiz': {'value': factors['quiz_avg'],
                 'status': get_factor_status('quiz_avg', factors['quiz_avg']),
                 'thresholds': {'High':'<60','Medium':'60–69','Low':'≥70'}},
        'activity_gap': {'value': factors['days_since_last_activity'],
                         'status': get_factor_status('days_since_last_activity', factors['days_since_last_activity']),
                         'thresholds': {'High':'>30','Medium':'8–30','Low':'≤7'}},
        'mental_health': {'value': factors['mental_health_score'],
                          'status': get_factor_status('mental_health_score', factors['mental_health_score']),
                          'thresholds': {'High':'<5','Medium':'5–6','Low':'≥7'}},
        'financial': {'value': s.financial_status,
                      'status':'High' if s.financial_status=='Unstable' else 'Low',
                      'thresholds': {'High':'Unstable','Low':'Stable/Scholarship'}},
        'extracurricular': {'value': 'Yes' if s.extracurricular_activity else 'No',
                            'status': 'Low' if s.extracurricular_activity else 'Medium',
                            'thresholds': {'Low':'Yes','Medium':'No'}}
    }
    return jsonify({"factors": factors, "legend": legend})

# Individual prediction with locked thresholds
@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    try:
        latest = ModelMetrics.query.order_by(ModelMetrics.created_at.desc()).first()
        if not latest or not os.path.exists(app.config['MODEL_PATH']):
            return jsonify({"error": "Model not trained yet"}), 503
        with open(app.config['MODEL_PATH'], 'rb') as f:
            model = pickle.load(f)

        payload = request.get_json(force=True)
        features = {
            'attendance': float(payload['attendance']),
            'gpa': float(payload['gpa']),
            'assignment_avg': float(payload['assignment_avg']),
            'quiz_avg': float(payload['quiz_avg']),
            'days_since_last_activity': float(payload['days_since_last_activity']),
            'financial_stable': 1 if payload.get('financial_status') == 'Stable' else 0,
            'financial_unstable': 1 if payload.get('financial_status') == 'Unstable' else 0,
            'mental_health_score': int(payload.get('mental_health_score', 6)),
            'extracurricular_activity': 1 if payload.get('extracurricular_activity', False) else 0
        }
        X = pd.DataFrame([features])
        proba = model.predict_proba(X)[0]
        labels = list(model.named_steps['classifier'].classes_)
        prob_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

        if os.path.exists(app.config['ARTIFACTS_PATH']):
            with open(app.config['ARTIFACTS_PATH'],'r') as f:
                art = json.load(f)
            locked = art.get('locked_thresholds', {})
            thr_high = locked.get('alt', {}).get('high_target_recall_0.80',
                                                locked.get('high_f1', 0.6))
            thr_low  = locked.get('low_f1', 0.6)
        else:
            thr_high, thr_low = 0.6, 0.6
        band = _apply_locked_bands(prob_dict, thr_high, thr_low)

        risk_score = round(prob_dict.get('High',0)*100 + prob_dict.get('Medium',0)*50, 2)

        session['last_prediction'] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": band, "probabilities": prob_dict, "risk_score": risk_score
        }

        try:
            clf = model.named_steps['classifier'].base_estimator if hasattr(model.named_steps['classifier'],'base_estimator') else model.named_steps['classifier']
            importances = _extract_feature_importances_from_classifier(clf)
            if importances is not None:
                feat_names = list(X.columns)
                feature_importance = dict(zip(feat_names, map(float, importances[:len(feat_names)])))
            else:
                feature_importance = json.loads(latest.feature_importance)
        except Exception:
            feature_importance = json.loads(latest.feature_importance)

        risk_factors = {
            'attendance': {'value': float(features['attendance']),
                           'status': get_factor_status('attendance', features['attendance']),
                           'thresholds': {'High':'<70','Medium':'70–79','Low':'≥80'}},
            'gpa': {'value': float(features['gpa']),
                    'status': get_factor_status('gpa', features['gpa']),
                    'thresholds': {'High':'<2.5','Medium':'2.5–2.9','Low':'≥3.0'}},
            'assignment_avg': {'value': float(features['assignment_avg']),
                               'status': get_factor_status('assignment_avg', features['assignment_avg']),
                               'thresholds': {'High':'<60','Medium':'60–69','Low':'≥70'}},
            'quiz_avg': {'value': float(features['quiz_avg']),
                         'status': get_factor_status('quiz_avg', features['quiz_avg']),
                         'thresholds': {'High':'<60','Medium':'60–69','Low':'≥70'}},
            'activity_gap': {'value': float(features['days_since_last_activity']),
                             'status': get_factor_status('days_since_last_activity', features['days_since_last_activity']),
                             'thresholds': {'High':'>30','Medium':'8–30','Low':'≤7'}},
            'mental_health': {'value': int(features['mental_health_score']),
                              'status': get_factor_status('mental_health_score', features['mental_health_score']),
                              'thresholds': {'High':'<5','Medium':'5–6','Low':'≥7'}},
            'financial': {'value': payload.get('financial_status'),
                          'status': 'High' if payload.get('financial_status')=='Unstable' else 'Low',
                          'thresholds': {'High':'Unstable','Low':'Stable/Scholarship'}},
            'extracurricular': {'value': 'Yes' if payload.get('extracurricular_activity', False) else 'No',
                                'status': 'Low' if payload.get('extracurricular_activity', False) else 'Medium',
                                'thresholds': {'Low':'Yes','Medium':'No'}}
        }

        return jsonify({
            "prediction": band,
            "probabilities": prob_dict,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "locked_thresholds": {"high": thr_high, "low": thr_low},
            "model_meta": {
                "accuracy": latest.accuracy, "precision": latest.precision,
                "recall": latest.recall, "last_trained": latest.created_at.isoformat()
            },
            "feature_importance": feature_importance
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/last-prediction')
def last_prediction():
    payload = session.get('last_prediction')
    if not payload: return jsonify({"message":"No recent prediction."}), 204
    return jsonify(payload)

# Manual training endpoint
@app.route('/api/train-now', methods=['POST'])
def train_now():
    _require_admin()
    try:
        artifacts = train_dropout_model()
        return jsonify({"ok": True, "artifacts": artifacts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Static files and uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['ADMIN_PICS'], filename)

@app.route('/favicon.ico')
def favicon(): return ('', 204)

@app.route('/<path:filename>')
def serve_static(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    # allow html too
    if ext in ('css','js','png','jpg','jpeg','gif','ico','svg','json','txt','map','html'):
        return send_from_directory('.', filename)
    abort(404)
    
#Admin Dept Routes
@app.route('/admin-departments.html')
def admin_departments_html():
    if not session.get('admin_logged_in'):
        return redirect(url_for('serve_admin_login_html'))
    admin = Admin.query.filter(func.lower(Admin.email) == (session.get('admin_email') or '').strip().lower()).first()
    return render_template('admin-departments.html', current_user=admin)

# ---- Departments ----
@app.route('/api/admin/departments', methods=['GET', 'POST'])
def api_admin_departments():
    _require_admin()

    if request.method == 'GET':
        rows = (Department.query
                .order_by(Department.slug.asc())
                .all())
        out = []
        for d in rows:
            out.append({
                "id": d.id,
                "slug": d.slug,
                "name": d.name,
                "staff_count": len(d.staff)
            })
        return jsonify(out)

    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    slug_in = data.get('slug') or data.get('department') or ''
    slug = _slugify(slug_in)
    staff_name = (data.get('staff_name') or '').strip()
    staff_email = (data.get('staff_email') or data.get('email') or '').strip().lower()
    staff_password = (data.get('staff_password') or data.get('password') or '')

    if not slug:
        return jsonify({"ok": False, "error": "Department slug is required"}), 400
    if Department.query.filter_by(slug=slug).first():
        return jsonify({"ok": False, "error": "Department already exists"}), 409

    dept = Department(slug=slug, name=data.get('name'))
    db.session.add(dept)
    db.session.flush()  # get dept.id

    staff_payload = None
    if staff_name and staff_email and staff_password:
        if Staff.query.filter_by(email=staff_email).first():
            db.session.rollback()
            return jsonify({"ok": False, "error": "Staff email already exists"}), 409

        staff = Staff(
            name=staff_name,
            email=staff_email,
            password_hash=generate_password_hash(staff_password),
            department_id=dept.id
        )
        db.session.add(staff)
        staff_payload = {"id": staff.id, "name": staff.name, "email": staff.email}

    db.session.commit()
    return jsonify({
        "ok": True,
        "department": {"id": dept.id, "slug": dept.slug, "name": dept.name},
        "staff": staff_payload
    }), 201

@app.route('/api/admin/departments/<int:dept_id>', methods=['PUT', 'DELETE'])
def api_admin_department_modify(dept_id):
    _require_admin()
    d = Department.query.get_or_404(dept_id)

    if request.method == 'DELETE':
        db.session.delete(d)
        db.session.commit()
        return jsonify({"ok": True})

    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
    new_slug = _slugify(data.get('slug') or data.get('new_slug') or d.slug)
    new_name = (data.get('name') or d.name)
    if new_slug != d.slug and Department.query.filter_by(slug=new_slug).first():
        return jsonify({"ok": False, "error": "Slug already in use"}), 409
    d.slug = new_slug
    d.name = new_name
    db.session.commit()
    return jsonify({"ok": True, "department": {"id": d.id, "slug": d.slug, "name": d.name}})

# ---- Staff ----
@app.route('/api/admin/staff', methods=['GET', 'POST'])
def api_admin_staff():
    _require_admin()

    if request.method == 'GET':
        dept_slug = request.args.get('dept')
        q = Staff.query
        if dept_slug:
            dep = Department.query.filter_by(slug=_slugify(dept_slug)).first()
            if not dep:
                return jsonify([]) 
            q = q.filter_by(department_id=dep.id)
        rows = q.order_by(Staff.id.desc()).all()
        return jsonify([{
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "department": {"id": s.department.id, "slug": s.department.slug, "name": s.department.name}
        } for s in rows])

    # POST create a staff under a department
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    dept_slug = _slugify(data.get('dept') or data.get('department') or '')
    dept_id = data.get('department_id')

    dep = None
    if dept_id:
        dep = Department.query.get(dept_id)
    elif dept_slug:
        dep = Department.query.filter_by(slug=dept_slug).first()

    if not dep:
        return jsonify({"ok": False, "error": "Valid department required"}), 400

    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not all([name, email, password]):
        return jsonify({"ok": False, "error": "Name, email and password are required"}), 400
    if Staff.query.filter_by(email=email).first():
        return jsonify({"ok": False, "error": "Staff email already exists"}), 409

    staff = Staff(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        department_id=dep.id
    )
    db.session.add(staff); db.session.commit()
    return jsonify({"ok": True, "staff": {
        "id": staff.id, "name": staff.name, "email": staff.email,
        "department": {"id": dep.id, "slug": dep.slug, "name": dep.name}
    }}), 201


@app.route('/api/admin/staff/<int:staff_id>', methods=['PUT', 'DELETE'])
def api_admin_staff_modify(staff_id):
    _require_admin()
    s = Staff.query.get_or_404(staff_id)

    if request.method == 'DELETE':
        db.session.delete(s); db.session.commit()
        return jsonify({"ok": True})

    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    new_name = data.get('name', s.name)
    new_email = (data.get('email') or s.email).strip().lower()
    new_password = data.get('password')
    new_dept_slug = data.get('dept') or data.get('department')
    new_dept_id = data.get('department_id')
    if new_dept_id or new_dept_slug:
        dep = Department.query.get(new_dept_id) if new_dept_id else Department.query.filter_by(slug=_slugify(new_dept_slug)).first()
        if not dep:
            return jsonify({"ok": False, "error": "Target department not found"}), 400
        s.department_id = dep.id

    if new_email != s.email and Staff.query.filter_by(email=new_email).first():
        return jsonify({"ok": False, "error": "Email already in use"}), 409

    s.name = new_name
    s.email = new_email
    if new_password:
        s.password_hash = generate_password_hash(new_password)

    db.session.commit()
    return jsonify({"ok": True})

# Staff authentication 
def _current_staff():
    """Returns the logged-in Staff row or None."""
    if not session.get('staff_logged_in'):
        return None
    email = (session.get('staff_email') or '').strip().lower()
    if not email:
        return None
    return Staff.query.filter(func.lower(Staff.email) == email).first()

def _dept_profile_page(slug: str) -> str:
    slug = (slug or '').strip().lower()
    mapping = {
        'academic': '/dept-academic.html',
        'finance': '/dept-finance.html',
        'mental_health': '/dept-mental-health.html',
        'mental-health': '/dept-mental-health.html',  
    }
    return mapping.get(slug, '/admin-dashboard')

@app.route('/staff/login', methods=['POST'])
def staff_login():
    # accept form or JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
    else:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

    s = Staff.query.filter(func.lower(Staff.email) == email).first()
    if not s or not check_password_hash(s.password_hash, password):
        # JSON if ajax, otherwise redirect back with flash
        if request.is_json:
            return jsonify({"ok": False, "error": "Invalid email or password"}), 401
        flash('Invalid email or password', 'danger')
        return redirect(url_for('auth_page'))

    # success: mark session
    session['staff_logged_in'] = True
    session['staff_email'] = s.email

    # JSON for SPA, or redirect for classic form posts
    redirect_url = _dept_profile_page(s.department.slug if s.department else '')
    if request.is_json:
        return jsonify({
            "ok": True,
            "staff": {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "department": {
                    "id": s.department.id,
                    "slug": s.department.slug,
                    "name": s.department.name
                }
            },
            "redirect": redirect_url
        })
    return redirect(redirect_url)

@app.route('/staff/logout')
def staff_logout():
    session.pop('staff_logged_in', None)
    session.pop('staff_email', None)
    return redirect(url_for('auth_page'))

@app.route('/api/staff/me', methods=['GET', 'PUT'])
def api_staff_me():
    """Single endpoint for staff self profile (GET + PUT)."""
    s = _current_staff()
    if not s:
        return jsonify({"error": "Not authenticated"}), 401

    if request.method == 'GET':
        return jsonify({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "profile_photo": s.profile_photo,
            "department": {"id": s.department.id, "slug": s.department.slug, "name": s.department.name}
        })

    # PUT (JSON or multipart)
    if request.content_type and 'multipart/form-data' in request.content_type:
        form = request.form
        s.name = form.get('name', s.name)
        new_email = (form.get('email') or s.email).strip().lower()
        if new_email != s.email and Staff.query.filter_by(email=new_email).first():
            return jsonify({"ok": False, "error": "Email already in use"}), 409
        s.email = new_email
        if form.get('password'):
            s.password_hash = generate_password_hash(form.get('password'))
        if 'photo' in request.files:
            f = request.files['photo']
            if f and f.filename:
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['ADMIN_PICS'], filename)
                f.save(path)
                s.profile_photo = f'/uploads/{filename}'
        db.session.commit()
        session['staff_email'] = s.email
        return jsonify({"ok": True})

    data = request.get_json(force=True) or {}
    s.name = data.get('name', s.name)
    new_email = (data.get('email') or s.email).strip().lower()
    if new_email != s.email and Staff.query.filter_by(email=new_email).first():
        return jsonify({"ok": False, "error": "Email already in use"}), 409
    s.email = new_email
    if data.get('password'):
        s.password_hash = generate_password_hash(data['password'])
    db.session.commit()
    session['staff_email'] = s.email
    return jsonify({"ok": True})

def _require_staff_for(slug_expected: str):
    s = _current_staff()
    if not s:
        return None, redirect(url_for('auth_page'))
    dept_slug = (s.department.slug if s.department else '').lower()
    # if staffs are logged in to a different department, send them to their own page
    if slug_expected and dept_slug not in (slug_expected, slug_expected.replace('_', '-')):
        return s, redirect(_dept_profile_page(dept_slug))
    return s, None

@app.route('/dept-academic.html')
def dept_academic_html():
    s, r = _require_staff_for('academic')
    if r: return r
    return render_template('dept-academic.html', current_staff=s)

@app.route('/dept-finance.html')
def dept_finance_html():
    s, r = _require_staff_for('finance')
    if r: return r
    return render_template('dept-finance.html', current_staff=s)

@app.route('/dept-mental-health.html')
def dept_mental_health_html():
    s, r = _require_staff_for('mental_health')
    if r: return r
    return render_template('dept-mental-health.html', current_staff=s)

@app.route('/dept/<slug>')
def dept_shortcut(slug):
    return redirect(_dept_profile_page(slug))

def _staff_required_json():
    """Return (staff, error_response_or_None)."""
    s = _current_staff()
    if not s:
        return None, (jsonify({"error": "Not authenticated"}), 401)
    return s, None

def _json_user_brief(u: User):
    if not u: return None
    return {"id": u.id, "name": u.name, "email": u.email}

@app.route('/api/staff/users', methods=['GET'])
def api_staff_users():
    s = _current_staff()
    if not s:
        return jsonify({"error": "Not authenticated"}), 401

    term = (request.args.get('q') or '').strip().lower()
    q = User.query
    if term:
        like = f"%{term}%"
        q = q.filter(or_(func.lower(User.name).like(like),
                         func.lower(User.email).like(like)))
    users = q.order_by(User.id.desc()).all()

    out = []
    for u in users:
        last = (PredictionHistory.query
                .filter_by(user_id=u.id)
                .order_by(PredictionHistory.created_at.desc())
                .first())
        if last:
            raw = last.risk_score or 0.0
            risk100 = raw * 100.0 if raw <= 1.0 else raw
            last_band = last.prediction
            last_risk = round(float(risk100), 1)
            last_time = last.created_at.isoformat()
        else:
            last_band = None; last_risk = None; last_time = None

        out.append({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "profile_photo": u.profile_photo or "/uploads/default.png",
            "last_band": last_band,
            "last_risk": last_risk,
            "last_time": last_time
        })
    return jsonify(out)

def _json_staff_public(s):
    d = getattr(s, "department", None)
    return {
        "id": s.id,
        "name": s.name,
        "email": s.email,
        "profile_photo": getattr(s, "profile_photo", None),
        "department": None if not d else {
            "id": d.id,
            "slug": d.slug,
            "name": d.name,
        },
    }
# --- LIST STAFF  ---
@app.get("/api/department/staff")
def api_department_staff():
    s = _current_staff()  
    scope = (request.args.get("dept") or "").strip().lower() 

    q = Staff.query
    if scope in ("", "same"):                     
        q = q.filter(Staff.department_id == s.department_id)
    elif scope != "all":                          
        dep = Department.query.filter(func.lower(Department.slug) == scope).first()
        if not dep: return jsonify([])
        q = q.filter(Staff.department_id == dep.id)

    rows = q.order_by(func.lower(Staff.name)).all()
    return jsonify([_json_staff_public(x) for x in rows])


@app.get("/api/department/admins")
def api_department_admins():
    s = _current_staff()  # require a staff session
    if not s:
        return jsonify({"error": "Not authenticated"}), 401
    rows = Admin.query.order_by(func.lower(Admin.name)).all()
    out = []
    for a in rows:
        out.append({
            "id": a.id,
            "name": a.name or "Admin",
            "email": a.email,
            "department": {"slug": "administration", "name": a.department or "Administration"}
        })
    return jsonify(out)


@app.route("/api/emails", methods=["GET", "POST"])
def api_emails():
    # use override-aware resolver
    role, me = _whoami_from_request()

    # SEND
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        subject = (data.get("subject") or data.get("topic") or "Message").strip()
        body    = (data.get("body") or "").strip()
        to_raw  = data.get("to") or data.get("to_email") or data.get("to_emails")
        recipients = [to_raw] if isinstance(to_raw, str) else [str(x) for x in (to_raw or [])]
        recipients = [x.strip().lower() for x in recipients if x and "@" in x]

        if not recipients:
            return jsonify({"error": "Recipient required"}), 400
        if not body:
            return jsonify({"error": "Message body is required"}), 400

        for addr in recipients:
            db.session.add(EmailLog(
                from_role=role, from_email=me, to_email=addr,
                subject=subject, body=body
            ))
        db.session.commit()

        print(f"[mail] {role} {me} to {', '.join(recipients)} : {subject}\n{body}\n", flush=True)
        return jsonify({"ok": True, "sent": len(recipients)})

    # LIST
    box = (request.args.get("box") or "inbox").strip().lower()  # inbox|sent|trash
    q    = (request.args.get("q") or "").strip().lower()
    limit = max(1, min(int(request.args.get("limit", 200)), 500))

    query = EmailLog.query
    me_l = me.lower()

    if box == "sent":
        query = query.filter(
            func.lower(EmailLog.from_email) == me_l,
            EmailLog.deleted_by_sender == False
        )
    elif box == "trash":
        query = query.filter(
            or_(
                and_(func.lower(EmailLog.from_email) == me_l, EmailLog.deleted_by_sender == True),
                and_(func.lower(EmailLog.to_email)   == me_l, EmailLog.deleted_by_recipient == True),
            )
        )
    else:  # inbox
        query = query.filter(
            func.lower(EmailLog.to_email) == me_l,
            EmailLog.deleted_by_recipient == False
        )

    if q:
        like = f"%{q}%"
        query = query.filter(or_(func.lower(EmailLog.subject).like(like),
                                 func.lower(EmailLog.body).like(like)))

    rows = (query.order_by(EmailLog.created_at.desc())
                 .limit(limit)
                 .all())
    return jsonify([_email_to_json(r, me) for r in rows])

@app.route("/api/emails/<int:mid>", methods=["GET", "DELETE"])
def api_email_item(mid: int):
    # use override-aware resolver
    role, me = _whoami_from_request()
    me_l = me.lower()
    r = EmailLog.query.get_or_404(mid)

    am_sender = (r.from_email or "").lower() == me_l
    am_rcpt   = (r.to_email or "").lower() == me_l
    if not (am_sender or am_rcpt):
        return jsonify({"error": "Forbidden"}), 403

    if request.method == "GET":
        # Auto-mark read if recipient opens
        if am_rcpt and not r.read_by_recipient:
            r.read_by_recipient = True
            db.session.commit()
        return jsonify(_email_to_json(r, me))

    # DELETE - move to trash for this viewer; if both deleted 
    if am_sender:
        r.deleted_by_sender = True
    if am_rcpt:
        r.deleted_by_recipient = True

    hard_deleted = False
    if r.deleted_by_sender and r.deleted_by_recipient:
        db.session.delete(r)
        hard_deleted = True

    db.session.commit()
    return jsonify({"ok": True, "hard_deleted": hard_deleted})

@app.route("/api/emails/<int:mid>/restore", methods=["PUT"])
def api_email_restore(mid: int):
    # use override-aware resolver
    role, me = _whoami_from_request()
    me_l = me.lower()
    r = EmailLog.query.get_or_404(mid)

    am_sender = (r.from_email or "").lower() == me_l
    am_rcpt   = (r.to_email or "").lower() == me_l
    if not (am_sender or am_rcpt):
        return jsonify({"error": "Forbidden"}), 403

    if am_sender:
        r.deleted_by_sender = False
    if am_rcpt:
        r.deleted_by_recipient = False

    db.session.commit()
    return jsonify({"ok": True})

@app.route("/api/emails/<int:mid>/read", methods=["PUT"])
def api_email_mark_read(mid: int):
    # use override-aware resolver
    role, me = _whoami_from_request()
    me_l = me.lower()
    r = EmailLog.query.get_or_404(mid)
    if (r.to_email or "").lower() != me_l:
        return jsonify({"error": "Only recipient can mark read"}), 403
    r.read_by_recipient = True
    db.session.commit()
    return jsonify({"ok": True})

@app.post("/api/send-email")
def api_send_email_alias():
    return api_emails()

@app.post("/api/staff/send_email")
def api_send_email_legacy2():
    return api_emails()

@app.route("/api/user/emails", methods=["GET", "POST"])
def api_user_emails_alias():
    # delegates to /api/emails for list & send
    return api_emails()

@app.route("/api/user/emails/<int:mid>", methods=["GET", "DELETE"])
def api_user_email_item_alias(mid: int):
    return api_email_item(mid)

@app.route("/api/user/emails/<int:mid>/restore", methods=["PUT"])
def api_user_email_restore_alias(mid: int):
    return api_email_restore(mid)

@app.route("/api/user/emails/<int:mid>/read", methods=["PUT"])
def api_user_email_mark_read_alias(mid: int):
    return api_email_mark_read(mid)

@app.post("/api/user/email/send")
def api_user_email_send_alias():
    # POST body: { to, subject, body }
    return api_emails()

@app.get("/api/user/contacts/staff")
def api_user_contacts_staff():
    if not session.get("user_logged_in"):
        return jsonify({"error": "Not authenticated"}), 401

    scope = (request.args.get("dept") or "all").strip().lower()
    q = Staff.query.join(Department, isouter=True)

    if scope not in ("", "all"):
        if scope in ("same", "current"):
            # try to filter by the user's own department when present
            user = User.query.filter(func.lower(User.email) ==
                                     (session.get("user_email") or "").strip().lower()).first()
            if user and user.department:
                q = q.filter(func.lower(Department.slug) == user.department.lower())
        else:
            q = q.filter(func.lower(Department.slug) == scope)

    rows = q.order_by(func.lower(Staff.name)).all()
    out = []
    for s in rows:
        d = getattr(s, "department", None)
        out.append({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "department": (None if not d else {"slug": d.slug, "name": d.name})
        })
    return jsonify(out)

@app.get("/api/user/contacts/admins")
def api_user_contacts_admins():
    if not session.get("user_logged_in"):
        return jsonify({"error": "Not authenticated"}), 401
    rows = Admin.query.order_by(func.lower(Admin.name)).all()
    return jsonify([{
        "id": a.id,
        "name": a.name or "Admin",
        "email": a.email,
        "department": {"slug": "administration", "name": a.department or "Administration"}
    } for a in rows])

# Entrypoint
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        basedir = os.path.abspath(os.path.dirname(__file__))
        app.config.setdefault('SQLALCHEMY_DATABASE_URI', f"sqlite:///{os.path.join(basedir, 'app.db')}")
        app.config['MODEL_PATH'] = app.config.get('MODEL_PATH') or os.path.join(basedir, 'dropout_model.pkl')
        app.config['ARTIFACTS_PATH'] = app.config.get('ARTIFACTS_PATH') or os.path.join(basedir, 'model_artifacts.json')
        db_path = os.path.join(basedir, 'app.db')
        model_path = app.config['MODEL_PATH']
        artifacts_path = app.config['ARTIFACTS_PATH']
        try:
            from sqlalchemy import text, inspect
            insp = inspect(db.engine)
            if insp.has_table('staff'):
                res = db.session.execute(text("PRAGMA table_info(staff)"))
                cols = []
                for row in res:
                    # SQLAlchemy 2.x Row: use _mapping; fallback to positional index for older versions
                    mapping = getattr(row, "_mapping", None)
                    cols.append(mapping['name'] if mapping and 'name' in mapping else row[1])
                if 'profile_photo' not in cols:
                    db.session.execute(text("ALTER TABLE staff ADD COLUMN profile_photo VARCHAR(200)"))
                    db.session.commit()
        except Exception as mig_err:
            print(f"[warn] Skipped staff.profile_photo migration check: {mig_err}", flush=True)
        
        print(
            "\nUsing files:\n"
            f" • DB:       {db_path}\n"
            f" • Model:    {model_path}\n"
            f" • Artifacts:{artifacts_path}\n",
            flush=True
        )
        try:
            from sqlalchemy import text, inspect
            insp = inspect(db.engine)
            if insp.has_table('email_log'):
                res = db.session.execute(text("PRAGMA table_info(email_log)"))
                cols = []
                for row in res:
                    mapping = getattr(row, "_mapping", None)
                    cols.append(mapping['name'] if mapping and 'name' in mapping else row[1])
                if 'read_by_recipient' not in cols:
                    db.session.execute(text("ALTER TABLE email_log ADD COLUMN read_by_recipient BOOLEAN DEFAULT 0"))
                if 'deleted_by_sender' not in cols:
                    db.session.execute(text("ALTER TABLE email_log ADD COLUMN deleted_by_sender BOOLEAN DEFAULT 0"))
                if 'deleted_by_recipient' not in cols:
                    db.session.execute(text("ALTER TABLE email_log ADD COLUMN deleted_by_recipient BOOLEAN DEFAULT 0"))
                db.session.commit()
        except Exception as mig_err:
            print(f"[warn] Skipped email_log migration check: {mig_err}", flush=True)

        try:
            need_seed = (not os.path.exists(db_path)) or (Student.query.count() == 0)
        except Exception:
            need_seed = True

        if need_seed:
            print("Generating enhanced student data...", flush=True)
            generate_random_students(1000)
            print(f"Generated {Student.query.count()} students.", flush=True)

        # Ensure a default Admin exists
        if Admin.query.filter_by(email='admin@example.com').first() is None:
            db.session.add(Admin(
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                name='Administrator'
            ))
            db.session.commit()

        # Train a model once if none on disk
        if not os.path.exists(model_path):
            print("No trained model found, starting initial training...", flush=True)
            try:
                train_dropout_model()
                print("Initial training complete.", flush=True)
            except Exception as e:
                print(f"Initial training failed: {e}", flush=True)
        else:
            print("Model file found, skipping auto-training.", flush=True)

    host, port = '127.0.0.1', 5000
    print("\nOpen your app:")
    print(f" • Index page:  http://{host}:{port}/index.html")
    if not IMB_READY:
        print(" imblearn not found, install with `pip install imbalanced-learn` to enable oversampling.")
    if not SHAP_READY:
        print("shap not found, install with `pip install shap` for explainability.")
    app.run(host=host, port=port, debug=True, use_reloader=False)
