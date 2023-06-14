const category = document.getElementById("high");
const upper = document.getElementsByClassName("top")[0];
const upperButton = document.getElementById("button");

function handleClick() {
  const clickedClass = "top";
  if (upper.classList.contains(clickedClass)) {
    upper.classList.remove(clickedClass);
    upperButton.style.transform = "scaleY(-1)";
    upperButton.style.transition = ".3s";
  } else {
    upper.classList.add(clickedClass);
    upperButton.style.transform = "scaleY(1)";
    upperButton.style.transition = ".3s";
  }
}

category.addEventListener("click", handleClick);

const category1 = document.getElementById("low");
const lower = document.getElementsByClassName("pants")[0];
const lowerButton = document.getElementById("button1");

function handleClick1() {
  const clickedClass = "pants";
  if (lower.classList.contains(clickedClass)) {
    lower.classList.remove(clickedClass);
    lowerButton.style.transform = "scaleY(-1)";
    lowerButton.style.transition = ".3s";
  } else {
    lower.classList.add(clickedClass);
    lowerButton.style.transform = "scaleY(1)";
    lowerButton.style.transition = ".3s";
  }
}

category1.addEventListener("click", handleClick1);

/*ai get img*/
function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $(".image-upload-wrap").hide();

      $(".file-upload-image").attr("src", e.target.result);
      $(".file-upload-content").show();

      $(".image-title").html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);
  } else {
    removeUpload();
  }
}

function removeUpload() {
  $(".file-upload-input").replaceWith($(".file-upload-input").clone());
  $(".file-upload-content").hide();
  $(".image-upload-wrap").show();
}
$(".image-upload-wrap").bind("dragover", function () {
  $(".image-upload-wrap").addClass("image-dropping");
});
$(".image-upload-wrap").bind("dragleave", function () {
  $(".image-upload-wrap").removeClass("image-dropping");
});
