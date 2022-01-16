import io
import base64
from flask import Blueprint, render_template, request, flash, redirect
from .utils import allowed_file, find_y, find_x, loader
from .model import inference
from flask_login import login_required, current_user

views = Blueprint(
    "views",
    __name__,
)


@views.route("/home")
@login_required
def home():
    return render_template("home.html", user=current_user)


@views.route("/info")
@login_required
def info():
    return render_template("info.html", user=current_user)


@views.route("/appointment")
@login_required
def appointment():
    return render_template("appointment.html", user=current_user)


@views.route("/diagnose", methods=["GET", "POST"])
@login_required
def diagnose():
    if request.method == "POST":
        f = request.files["file"]
        if f.filename == "":
            flash("Please select a file.")
            return redirect(request.url)
        elif not allowed_file(f.filename):
            flash("Invalid file type.")
            return redirect(request.url)
        imgList = [loader(f)]
        flash("Upload successfully.")

        img_data = []
        for img in imgList:
            data = io.BytesIO()
            img.save(data, "JPEG")
            # convert to base64 in byte
            encoded_img_data = base64.b64encode(data.getvalue())
            # convert to base64 in utf-8
            decoded_img_data = encoded_img_data.decode("utf-8")
            img_data.append(decoded_img_data)
        out = inference(imgList)
        return render_template("diagnose.html", img_data=img_data, res=out, user=current_user)
    else:
        return render_template("diagnose.html", user=current_user)
