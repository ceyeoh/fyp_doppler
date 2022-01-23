from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .database import User
from flask_login import login_user, login_required, current_user, logout_user


auth = Blueprint(
    "auth",
    __name__,
)


@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        firstname = request.form.get("firstname")
        email = request.form.get("email")
        password0 = request.form.get("password0")
        password1 = request.form.get("password1")
        user = User.query.filter_by(email=email).first()
        if user:
            flash("User already exists!", category="error")
        elif len(firstname) < 2:
            flash("Name must be greater than 1 character!", category="error")
        elif len(email) < 5:
            flash("Email must be greater than 4 characters!", category="error")
        elif len(password0) < 6:
            flash("Password must be at least 6 characters!", category="error")
        elif password0 != password1:
            flash("Passwords don't match!", category="error")
        else:
            new_user = User(
                firstname=firstname,
                email=email,
                password=generate_password_hash(password0, method="sha256"),
            )
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully!", category="succcess")
            return redirect(url_for("views.home"))
    return render_template("signup.html", user=current_user)


@auth.route("/", methods=["GET", "POST"])
@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash("Login successfully!", category="success")
                login_user(user, remember=True)
                return redirect(url_for("views.home"))
            else:
                flash("Incorrect password, try again!", category="error")
        else:
            flash("Email does not exist!", category="error")
    return render_template("login.html", user=current_user)


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
