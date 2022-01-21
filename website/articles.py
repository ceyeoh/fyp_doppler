from flask import Blueprint, render_template
from flask_login import login_required, current_user

articles = Blueprint(
    "articles",
    __name__,
)


@articles.route("/intro-fgr")
@login_required
def intro():
    return render_template("article-intro-fgr.html", user=current_user)


@articles.route("/causes-fgr")
@login_required
def causes():
    return render_template("article-causes-fgr.html", user=current_user)


@articles.route("/twinsrisk-fgr")
@login_required
def twinsrisk():
    return render_template("article-twinsrisk-fgr.html", user=current_user)


@articles.route("/symptoms-fgr")
@login_required
def symptoms():
    return render_template("article-symptoms-fgr.html", user=current_user)


@articles.route("/diagnosis-fgr")
@login_required
def diagnosis():
    return render_template("article-diagnosis-fgr.html", user=current_user)


@articles.route("/preventions-fgr")
@login_required
def preventions():
    return render_template("article-preventions-fgr.html", user=current_user)
