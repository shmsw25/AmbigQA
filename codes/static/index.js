(function() {

  var questions;
  var mode=0;
  var custom_message = ">>> Write your own question <<<";

  $( window ).init(function(){
		load();
	});

	function load(){
    sendAjax("/select", {}, (data) => {
      questions = data.questions;
			var dropdown = document.getElementById("question");
			for(var i=0; i<questions.length; i++){
				var opt = document.createElement("option");
				opt.value = parseInt(i);
        opt.id = "question-option-"+parseInt(i);
				opt.innerHTML = questions[i];
        dropdown.appendChild(opt);
      }

      $('.editOption').keyup(function () {
        var editText = $('.editOption').val();
        $('editable').val(editText);
        $('editable').html(editText);
      });

      $('.editOption').on('click', function () {
        $('#answer').html('')
      });

      $(".run").click(loadAnswer);

      $('.mode').click(function() {
        mode = parseInt($('.mode:checked').val());
        $('#answer').html('');
        if (mode === 0) {
          $('#select-question').show();
          $('#write-question').hide();
        } else {
          $('#select-question').hide();
          $('#write-question').show();
        }
      });

      /* Label Tooltip */
      $('.mode').mouseover(function(event){
        $('#mode-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
        if (parseInt(event.target.value)===0)
          $('#mode-tooltip').html("You can see example questions from NQ-open.");
        else
          $('#mode-tooltip').html("You can write your own questions.");
      });
      $('.mode').mouseout(function(){
        $('#mode-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');

      });

      /* Description Button */
      $('#description-button').click(function(){
        if ($('#description-button').html() === 'Show Me Details!') {
          $('#description').show();
          $('#description-button').html('Hide Details!');
        } else {
          $('#description').hide();
          $('#description-button').html('Show Me Details!');
        }
      })

		});
  }

  function loadAnswer(){
    var question_text = $('select#question option:selected').html();
    if (mode === 1) {
      question_text = $('.editOption').val();
      if (!(question_text.replace(/\s/g, '').length)) {
        alert('Please enter a non-empty question.');
        return;
      }
      if (!(paragraphs_text.replace(/\s/g, '').length)) {
        alert('Please enter a non-empty paragraph.');
        return;
      }
    }
    var k = parseInt($('#k').val());
    if (!Number.isInteger(k)) {
      alert('Please enter a valid number for # of answers.');
      return;
    }
    if (k<1 || k>100) {
      alert('Please enter a number between 1 and 100.');
      return;
    }
    console.log(k);
		document.getElementById("answer").innerHTML = "";
		document.getElementById("loading").style.display = "block";
		var data = {
      'question': question_text, 'k': k
    };
		sendAjax("/submit", data, (result) => {
			document.getElementById("loading").style.display = "none";
      var answer_field = document.getElementById('answer');
      for (var i=0; i<result.length; i++) {
        var metadata = `
          <span class='footnote-sm'><em>P(a|q)</em>=` + result[i]["softmax"]["joint"].toFixed(2) + `</span>
          <span class='footnote-sm'><em>P(a|p,q)</em>=` + result[i]["softmax"]["span"].toFixed(2) + `</span>
          <span class='footnote'><em>P(p|q)</em>=` + result[i]["softmax"]["passage"].toFixed(2) + `</span>
          <span class='footnote'>Retrieval rank: #` + (result[i]["passage_index"]+1) + `</span>`;
        var header = "<strong>" + result[i]["title"] + `</strong>` + metadata;
        answer_field.appendChild(getPanel(header, "... " + result[i]["passage"] + " ..."));
      }
    });
	}
  function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}

	function getPanel(heading_text, context_text){
		var div = document.createElement('div');
		div.className = "panel panel-default";
		var heading = document.createElement('div');
		heading.className = "panel-heading";
		heading.innerHTML = heading_text;
		var context = document.createElement('div');
		context.className = "panel-body";
		context.innerHTML = context_text;
		div.appendChild(heading);
		div.appendChild(context);
		return div
	}

  function getPanel2(heading_text, context_text, footer_text){
		var div = document.createElement('div');
		div.className = "panel panel-default";
		var heading = document.createElement('div');
    heading.className = "panel-heading my-heading";
    heading.style.width = "150px";
    heading.style.float = "left";
		heading.innerHTML = heading_text;
    var context = document.createElement('div');
    context.className = "panel-body";
    //context.style.float = "left";
		context.innerHTML = context_text;
    var footer = document.createElement('div');
    footer.className = "panel-heading";
    footer.style.width = "150px";
    footer.style.float = "right";
		footer.innerHTML = footer_text;


    div.appendChild(heading);
    div.appendChild(footer);
    div.appendChild(context);

    //heading.style.height = context.height;
    //footer.style.height = context.height;

    return div
	}


})();



