function readImage(input) {
    console.log(input.files);
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imgcontainer')
                .attr('src', e.target.result)
                .width(600)
                .height(400);
        };

        reader.readAsDataURL(input.files[0]);
    }
}