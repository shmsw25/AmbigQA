var APIKEY = "";
var SEARCHID = "";

var PAY_BASE = 10;
var PAY_DELTA = 10;
var PAY_MULTIQA = 60;
var currentPay = PAY_BASE+PAY_MULTIQA;
var cache = [];
var UNIQ_ANSWER_ID = 1001;
var current_index = 0;
var current_step = 1;
/*! js-cookie v3.0.0-beta.3 | MIT */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):(e=e||self,function(){var n=e.Cookies,r=e.Cookies=t();r.noConflict=function(){return e.Cookies=n,r}}())}(this,function(){"use strict";var e={read:function(e){return e.replace(/(%[\dA-F]{2})+/gi,decodeURIComponent)},write:function(e){return encodeURIComponent(e).replace(/%(2[346BF]|3[AC-F]|40|5[BDE]|60|7[BCD])/g,decodeURIComponent)}};function t(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)e[r]=n[r]}return e}return function n(r,o){function i(e,n,i){if("undefined"!=typeof document){"number"==typeof(i=t({},o,i)).expires&&(i.expires=new Date(Date.now()+864e5*i.expires)),i.expires&&(i.expires=i.expires.toUTCString()),n=r.write(n,e),e=encodeURIComponent(e).replace(/%(2[346B]|5E|60|7C)/g,decodeURIComponent).replace(/[()]/g,escape);var c="";for(var u in i)i[u]&&(c+="; "+u,!0!==i[u]&&(c+="="+i[u].split(";")[0]));return document.cookie=e+"="+n+c}}return Object.create({set:i,get:function(t){if("undefined"!=typeof document&&(!arguments.length||t)){for(var n=document.cookie?document.cookie.split("; "):[],o={},i=0;i<n.length;i++){var c=n[i].split("="),u=c.slice(1).join("=");'"'===u[0]&&(u=u.slice(1,-1));try{var f=e.read(c[0]);if(o[f]=r.read(u,f),t===f)break}catch(e){}}return t?o[t]:o}},remove:function(e,n){i(e,"",t({},n,{expires:-1}))},withAttributes:function(e){return n(this.converter,t({},this.attributes,e))},withConverter:function(e){return n(t({},this.converter,e),this.attributes)}},{attributes:{value:Object.freeze(o)},converter:{value:Object.freeze(r)}})}(e,{path:"/"})});

let ENCODETITLE = [" ", "%", "!", "#", "$", "&", "'", "(", ")", "*", "+", ",", "/", ":", ";", "=",
  "?", "@", "[", "]"];
let ENCODEURL = ["_", "%25", "%21", "%23", "%24", "%26", "%27", "%28", "%29",
  "%2A"< "%2B", "%2C", "%2F", "%3A", "%3B", "%3D", "%3F", "%40", "%5B", "%5D"];


/* instructions */

var EMPTYSTRING = "Please fill out all text boxes (or delete them).";
var EMPTYSTRING_SINGLEANSWER = "Please fill out all text boxes (or check 'Answer not found').";
var DUPLICATED_Q = "Questions should be different from each other.";
var DUPLICATED_A = "Answers should be different from each other.";
var SAME_Q = "Questions should be different from the prompt question.";
var ONEPAIR = "Please enter more than one pair (or check 'Single clear answer').";
var ZEROPAIR = "If there is no answer, check 'Single clear answer'."
var QUESTIONMARK = "Each question should end with a question mark ('?').";
var WRITEQUESTION = `If questions are empty, write questions that only allows the paired
    answer without any ambiguity.`;

var startTime;
var questionId;
var promptQuestion;
var promptAnnotations;
var answerId2questions = {};
var answerId2answers = {};
var promptAnnotations = [
  {"MultipleQAPairs": {"qaPairs": [
    {"question": "Which team has the most NCAA Division I Men's Ice Hockey Tournament appearances?", "answer": "Michigan and Minnesota"},
    {"question": "Which team has the most NCAA Division I Men's Basketball Tournament appearances?", "answer": "UCLA"}
  ]}},
  {"SingleAnswer": {"answer": "Kentucky"}},
  {"MultipleQAPairs": {"qaPairs": [
    {"question": "Which team has the most NCAA Division I tournament appearances?", "answer": "Kentucky Wildcats"},
    {"question": "Which team has the most NCAA Division II tournament appearances?", "answer": "Kentucky Wesleyan"}
  ]}}
];


function loadAll(INSTRUCTIONS){
  startTime = new Date().getTime();

  var prompt = JSON.parse($("#prompt").attr("value"));
  if ($('#taskKey').attr("value")==='"validation"') {
    prompt = prompt['generationPrompt'];
  } else if ($('#taskKey').attr("value")==='"final"') {
    console.log(prompt);
    prompt = prompt['validationPrompt']['generationPrompt'];
  }

  quesitonId = prompt['id']
  promptQuestion = prompt['question'];
  promptQuestion = promptQuestion.substring(0, 1).toUpperCase() + promptQuestion.substring(1, promptQuestion.length) + "?";
  $("#input-question").html(promptQuestion);
  $("#search-button").click(loadSearchResults);
  $('#search-query').keyup(function(event) {
      if (event.keyCode === 13) {
        event.preventDefault();
        if (event.stopPropagation!==undefined)
          event.stopPropagation();
        document.getElementById("search-button").click();
      }
  });
  $('#single-clear-answer-checkbox').prop('checked', false);
  $('#answer-not-found-checkbox').prop('checked', false);

  loadInstructions(INSTRUCTIONS);

  /* Prompt annotations */
  $('#prompt-annotations-header')
    .html('Annotations from each worker (Click to expand).')
    .mouseover(function(){
      this.style.textDecoration = "underline";
    })
    .mouseout(function(){
      this.style.textDecoration = "none";
    })
    .click(function(){
      if ($('#prompt-annotations').css('display')=='block') {
        $('#prompt-annotations').css('display', 'none');
        $('#prompt-annotations-header').html('Annotations from each worker (Click to expand).');
      } else {
        $('#prompt-annotations').css('display', 'block');
        $('#prompt-annotations-header').html('Annotations from each worker (Click to collapse).');
      }
    });
  $('#prompt-annotations').css('display', 'none');

  /* For search */
  $('#go-back-to-search-results').click(function() {
    $('#go-back-to-search-results').hide();
    $('#search-results').show();
    getAnnotations();
  });

  /* For default annotation */
  $('#go-back-to-default-annotation').click(function() {
    deleteAnnotationForms();
    $('.delete-pair-button').remove();
    setUnionAnnotations();
  });

  /* For checkboxes */
  $('#single-clear-answer-checkbox').change(function(){
    if (this.checked) {
      if ($('#answer-not-found-checkbox').prop('checked')) {
        $('#answer-not-found-checkbox').prop('checked', false);
        setPay(0);
      }
      deleteAnnotationFormsExceptOneAnswer();
      $('.delete-pair-button').remove();
      $('#add-pair-button').hide();
      $('#delete-pair-button').hide();
      $('#guide-box-response-type').html(INST_SINGLEANSWER);
      setPay(PAY_BASE + 10);
    } else {
      deleteAnnotationForms();
      setPay(PAY_BASE + PAY_MULTIQA);
      loadInitAnnotationForm();
      $('#add-pair-button').show()
      $('#delete-pair-button').show()
      $('#guide-box-response-type').html(INST_MULTIQAPAIRS);
    }
    getAnnotations();
  });
  $('#answer-not-found-checkbox').change(function(){
    if (this.checked) {
      deleteAnnotationForms();
      $('.delete-pair-button').remove();
      $('#single-clear-answer-checkbox').prop('checked', false);
      $('#add-pair-button').hide();
      $('#delete-pair-button').hide();
      setPay(PAY_BASE + 10);
      $('#guide-box-response-type').html(INST_NOANSWER);
    } else {
      setPay(PAY_BASE + PAY_MULTIQA);
      loadInitAnnotationForm();
      $('#add-pair-button').show();
      $('#delete-pair-button').show();
      $('#guide-box-response-type').html(INST_MULTIQAPAIRS);
    }
    getAnnotations();
  });
  $('#guide-box-response-type').html(INST_MULTIQAPAIRS);

  /* For buttons */
  $('#add-pair-button').click(function(){
    addAnnotationForm();
    getAnnotations();
  });
  $('.delete-pair-button').click(function(){
    deleteAnnotationForm();
  });
  $('#uw-checkbox').change(function(){
    if (this.checked) {
      alert(`If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research,
        You cannot participate in this job. Please return your HIT.`)
      $('#submit').prop('disabled', this.checked);
    } else {
      getAnnotations();
    }
  })
}

function loadInstructions(INSTRUCTIONS) {
  if (INSTRUCTIONS!==null) {
    /* For Instructions */
    $('#instruction-header')
      .html('Instructions (Click to collapse).')
      .mouseover(function(){
        this.style.textDecoration = "underline";
      })
      .mouseout(function(){
        this.style.textDecoration = "none";
      })
      .click(function(){
        if ($('#instruction-body').css('display')=='block') {
          // hiding instructions
          Cookies.set('hide-' + $('#taskKey').attr("value") + '-instruction', 'true', { expires: 7 })
          $('#instruction-body').css('display', 'none');
          $('#instruction-header').html('Instructions (Click to expand).');
        } else {
          // showing instructions
          Cookies.set('hide-' + $('#taskKey').attr("value") + '-instruction', 'false', { expires: 7 })
          document.cookie = "username=John Doe; expires=Thu, 18 Dec 2013 12:00:00 UTC";
          $('#instruction-body').css('display', 'block');
          $('#instruction-header').html('Instructions (Click to collapse).');
        }
      });
    if (Cookies.get('hide-' + $('#taskKey').attr("value") + '-instruction')==='true') {
      $('#instruction-body').css('display', 'none');
      $('#instruction-header').html('Instructions (Click to expand).');
    }
    $('.instructions-item').click(function() {
      $('.active').removeClass("active");
      $('#'+this.id).parent().addClass("active");
      $('#instructions').html(INSTRUCTIONS[this.id]);
    });
    $('#instructions').html(INSTRUCTIONS['instructions-overview']);
  }
}

function loadInitAnnotationForm() {
  if ($('#taskKey').attr("value")==='"validation"' && answerId2questions.length>1) {
    setUnionAnnotations()
    getAnnotations()
  } else {
    addAnnotationForm();
    addAnnotationForm();
    getAnnotations();
  }
}

function loadWhenWikipediaPageLoaded() {
  if ($('#taskKey').attr("value").startsWith('"generation') && $('.form-row').length===0) {
    setPay(PAY_BASE + PAY_MULTIQA);
    addAnnotationForm();
    addAnnotationForm();
    $('#checkboxes').show();
    $('#buttons').show();
  }
}

function sendAjax(url, data, handle){
  $.getJSON(url, data, function(response){
    handle(response.result);
  });
}

function setWidth() {
  $('#container').width($('#taskContent').width()-100);

  $(".panel").width($('#container').width());
  $(".input-group").width($('#container').width());
  $(".row").width($('#container').width());
  $(".wikipedia-box").width($('#container').width());
  $(".snippet").width($('#container').width()-10);
  if ($('#taskKey').attr("value").startsWith('"generation')) {
    $(".narrow-panel").width($('#container').width()*2/3);
    $(".narrow-input-group").width($('#container').width()*2/3);
    $(".narrow-snippet").width($('#container').width()*2/3-10);

    $(".narrow-wikipedia-box").width($('#container').width()*2/3);
    $(".panel-inline").width($('#container').width()/3-10);
    $("#feedback").width($('#container').width()/3-10);
  } else {
    $(".narrow-panel").width($('#container').width()/2-10);
    $(".narrow-input-group").width($('#container').width()/2-10);
    $(".narrow-snippet").width($('#container').width()/2-10);

    $(".narrow-wikipedia-box").width($('#container').width()/2-10);
    $(".panel-inline").width($('#container').width()/2-10);
    $("#feedback").width($('#container').width()/2-10);

    $('.col').width($('#container').width()/2-10);
  }
  //$('.form-control').width($('#container').width()/3-5);
}


function loadSearchResults(){
  $('#search-results').show();
  $('#go-back-to-search-results').hide();
  $('#search-results').html("");
  $('#wikipedia-box').html("");
  var query = $('#search-query').val();
  $.getJSON(
    "https://www.googleapis.com/customsearch/v1/siterestrict?key=" + APIKEY + "&cx=" + SEARCHID + "&q="+query,
    {},
    function(response){
      var searchResults = [];
      if (response.items===undefined) {
        $('#search-results').html("<span class='lg'>Error. Please try different search query.</span>");
        return;
      }
      response.items.forEach(function(d){
        console.assert(d.link.startsWith("https://en.wikipedia.org/wiki/"));
        var title = d.title;
        if (d.title.endsWith(' Wikipedia')) {
          title = title.substring(0, title.length-11);
        }
        var link = d.link.substring(0, 10) + ".m" + d.link.substring(10, d.link.length);
        var snippet = d.htmlSnippet.replace(/<br>/g, '');
        var extraClassName = " narrow-snippet"
        searchResults.push({'title': title, 'snippet': snippet});
        var item = $("<button type='button' class='snippet" + extraClassName + "'></button>")
          .html($("<span class='pretty p-icon p-curve p-rotate'" +
            "style='background-color: lightgray; border-radius: 2px'>" +
            title +"</span> <span class='snippet'>" + snippet + "</span>"))
          .mouseover(function(){
            this.style.textDecoration = 'underline';
          })
          .mouseout(function(){
            this.style.textDecoration = 'none';
          })
          .click(function(){
            $('#wikipedia-box').html("");
            $('#wikipedia-box').append("<iframe src='" + link +
              "' style='height: 450px; width: " +
              $('#wikipedia-box').width() + "px'></iframe>");
            $('.noprint').hide();
            $('#mw-navigation').hide();
            cache.push({'type': 'view', 'title': title});
            updateResponse();
            loadWhenWikipediaPageLoaded();
            $('#search-results').hide();
            $('#go-back-to-search-results').show();
            getAnnotations();
          });
        $('#search-results').append(item);
        $('#search-results').append($('<br />'));
      });
      cache.push({'type': 'search', 'query': query, 'results': searchResults});
      updateResponse();
    });
  getAnnotations();
}

function addAnnotationForm(question, answer){
  var questionDiv = $('<div></div>')
    .addClass('form-row')
    /*.append(`<div class="col">
      <input type="text" class="form-control question-form-control" placeholder="Question">
    </div>`);*/
     .append(`<div class="col">
      <textarea class="form-control question-form-control" placeholder="Question"></textarea>
    </div>`);

  var answerDiv = $('<div></div>')
    .addClass('form-row')
    .append(`
    <div class="col">
      <input type="text" class="form-control answer-form-control" placeholder="Answer">
    </div>`);
  $('#annotations').append(questionDiv);
  $('#annotations').append(answerDiv);
  var button = $('<button></button>')
    .attr('type', 'button')
    .addClass('btn btn-default delete-pair-button')
    .html('Delete pair')
    .click(function(){
      questionDiv.remove();
      answerDiv.remove();
      this.remove();
      setPay(currentPay - PAY_DELTA);
      getAnnotations();
    });
  $('#annotations').append(button);
  setPay(currentPay + PAY_DELTA);
  setUpAfterFormAdded();
  $('.form-control').width($('#container').width()/3-15);
}

function deleteAnnotationForm(){
  deleteQuestionForm();
  deleteAnswerForm();
  setPay(currentPay - PAY_DELTA);
  getAnnotations();
}

function deleteQuestionForm(){
  var questionForms = $('.question-form-control');
  questionForms[questionForms.length-1].remove();
}

function deleteAnswerForm(){
  var answerForms = $('.answer-form-control');
  answerForms[answerForms.length-1].remove();
}

function deleteAnnotationForms(){
  while($('.question-form-control').length>0) {
    deleteQuestionForm();
  }
  while($('.answer-form-control').length>0) {
    deleteAnswerForm();
  }
}
function deleteAnnotationFormsExceptOneAnswer(){
  var written = $('.answer-form-control');
  var union = ""
  for (var i=0; i<written.length; i++) {
    if ($.trim(written[i].value).length>0) {
      if (union.length>0)
        union += "|";
      union += written[i].value;
    }
  }
  while($('.question-form-control').length>0) {
    deleteQuestionForm();
  }
  if ($('.answer-form-control').length===0) {
    $('#annotations').append('<div class="form-row"><div class="col">' +
      '<input type="text" class="form-control answer-form-control" placeholder="Answer">' +
      '</div></div>');
    setPay(currentPay + PAY_DELTA);
  } else {
    while($('.answer-form-control').length>1) {
      deleteAnswerForm();
    }
  }
  $('.answer-form-control')[0].value = union;
}

function getStringOverlap(string1, string2) {
  var strings1 = string1.toLowerCase().split("|")
  var strings2 = string2.toLowerCase().split("|")
  for (var i=0; i<strings1.length; i++) {
    if (strings2.indexOf(strings1[i])>-1) {
      return true
    }
  }
  return false
}

function submitAnnotations() {
}

function getRelatedTitles(answer) {
  function _getRelatedTitle(answer) {
    let answerAppeared = cache.map(interaction => {
      if (interaction['type']==='singleAnswer') {
        return normalize(interaction['answer']).split('|').includes(answer);
      } else if (interaction['type']==='multipleQAs') {
        return interaction['qaPairs'].map(
          x => normalize(x['answer']).split('|').includes(answer)).includes(true);
      }
      return false
    });
    for (i=answerAppeared.indexOf(true); i>=0; i--) {
      if (cache[i]['type']==='view') {
        return $.trim(cache[i]['title']);
      }
    }
    return null;
  }
  var titles = [];
  normalize(answer).split('|').forEach(a => {
    let title = _getRelatedTitle(a);
    if (!(title===null || titles.includes(title))) {
      titles.push(title);
    }
  })
  return titles;
}

function getAnnotations() {

  console.log(cache);
  if ($('#taskKey').attr("value")!=='"final"' &&
      cache.map(interaction => interaction.type).indexOf("view")===-1) {
    $('#submit').prop('disabled', true);
    $('#check-my-response').prop('disabled', true);
    $('#validated-hint').html("You can submit after reading Wikipedia.");
    return
  }

  function _getDiffFromPrompt(questions) {
    function _getTokens(question) {
      return normalize(question).split(' ');
    }
    let promptTokens = _getTokens(promptQuestion);
    function _getDiff(tokens1, tokens2) {
      var diffs = [];
      for (i in tokens1) {
        let token = tokens1[i];
        if (!(tokens2.includes(token))) {
          diffs.push(token);
        }
      }
      return diffs;
    }
    let excludedCommonwords = ["the", "a", "that", "this", "which", "is", "are",
            "at", "on", "from", "in", "to", "of", "for",
            ".", "(", ")", "[", "]", "<", ">", "was", "were", "do", "does", "did"];
    function _getCommon(tokens1, tokens2) {
      var common = [];
      for (i in tokens1) {
        let token = tokens1[i];
        if (tokens2.includes(token) && !excludedCommonwords.includes(token)) {
          common.push(token);
        }
      }
      return common;
    }
    function _getUnion(tokens1, tokens2) {
      var union = tokens1;
      for (i in tokens2) {
        let token = tokens2[i];
        if (!union.includes(token)) {
          union.push(token);
        }
      }
      return union;
    }
    let questionTokens = questions.map(q => _getTokens(q));
    let added = questionTokens.map(tokens => _getDiff(tokens, promptTokens));
    let deleted = questionTokens.map(tokens => _getDiff(promptTokens, tokens));
    let addedCommon = (added.length===0) ? [] : added.reduce(_getCommon);
    let deletedCommon = (deleted.length===0) ? [] : deleted.reduce(_getCommon);
    let addedUnion = (added.length===0) ? [] : added.reduce(_getUnion);
    return {'added': addedCommon, 'deleted': deletedCommon, 'addedUnion': addedUnion}
  }

  var annotation;
  var validated = true;
  var validated_msg = "";

  if ($('#taskKey').attr("value")==='"qualification"' ||
      $('#taskKey').attr("value")==='"validation"'
  ) {
    if (current_step===1) {
      return;
    }
    //todo
    let answerId2answers = answer2question_list[current_index]["answerId2answers"];
    let answerId2questions = answer2question_list[current_index]["answerId2questions"];
    var annotation;
    var normalized_questions = [];
    if (Object.keys(answerId2answers).length===1) {
      annotation = {'type': 'singleAnswer',
        'answer': Object.values(answerId2answers)[0].join("|"),
        'titles': []
      };
    } else if (Object.keys(answerId2answers).length===0) {
      annotation = {'type': 'noAnswer'}
    } else {
      var qaPairs = [];
      Object.keys(answerId2answers).forEach(answerId => {
        let questions = answerId2questions[answerId].filter(x => !x.startsWith("[DISABLED]"));
        if (questions.length===0) {
          validated = false;
          validated_msg = WRITEQUESTION;
        }
        if (questions.map(q => q.endsWith('?')).includes(false)) {
          validated = false;
          validated_msg += QUESTIONMARK;
        }
        let curr_normalized_questions = questions.map(normalize);
        if (curr_normalized_questions.includes(normalize(promptQuestion))) {
          validated = false;
          validated_msg = "<br />" + SAME_Q;
        }
        curr_normalized_questions.forEach(question => {
          if (normalized_questions.includes(question)) {
            validated = false;
            validated_msg = DUPLICATED_Q;
          }
          normalized_questions.push(question);
        });
        qaPairs.push({'answer': answerId2answers[answerId].join("|"),
          "question": questions.join("|"), 'titles': []});
      })
      annotation = {'type': 'multipleQAs', 'qaPairs': qaPairs};
    }
  } else {
    var empty_question = false;
    var empty = false;
    var answerNotFound = $('#answer-not-found-checkbox').prop('checked');
    var singleClearAnswer = $('#single-clear-answer-checkbox').prop('checked');
    var written_annotations = $('.form-control');
    var annotation;
    if (answerNotFound) {
      annotation = {'type': 'noAnswer'};
    } else if (singleClearAnswer) {
      console.assert(written_annotations.length===1);
      if (written_annotations[0].value.length===0) {
        empty = true;
      }
      let answer = $.trim(written_annotations[0].value);
      annotation = {'type': 'singleAnswer', 'answer': answer, 'titles': getRelatedTitles(answer)};
    } else {
      console.assert(written_annotations.length%2===0);
      var qaPairs = [];
      var question;
      var answer;
      if (written_annotations.length===0) {
        validated = false;
        validated_msg = ZEROPAIR;
      } else if (written_annotations.length===2) {
        validated = false;
        validated_msg = ONEPAIR;
      }
      for (var i=0; i<written_annotations.length/2; i++) {
        question = written_annotations[2*i].value;
        answer = $.trim(written_annotations[2*i+1].value);
        if (question.length===0) {
          validated = false;
          validated_msg = ($('#taskKey').attr("value").startsWith('"generation')) ? EMPTYSTRING : WRITEQUESTION;
        }
        if (answer.length===0) {
          empty = true;
        }
        if (validated && qaPairs.map(d => getStringOverlap(d.question, question)).indexOf(true)>-1) {
          validated = false;
          validated_msg = DUPLICATED_Q;
        }
        qaPairs.push({'question': $.trim(question),
                      'answer': answer,
                      'titles': getRelatedTitles(answer)});
      }
      if (validated && qaPairs.map(d => getStringOverlap(d.question, promptQuestion)).indexOf(true)>-1) {
        validated = false;
        validated_msg = SAME_Q;
      }
      if (validated && qaPairs.map(d => d.question.endsWith("?")).indexOf(false)>-1) {
        validated = false;
        validated_msg = QUESTIONMARK;
      }
      annotation = {'type': 'multipleQAs', 'qaPairs': qaPairs};
    }
    if (validated && empty) {
      validated = false;
      validated_msg = (singleClearAnswer) ? EMPTYSTRING_SINGLEANSWER : EMPTYSTRING;
    }

    if (validated && $('#taskKey').attr("value")==='"generation"' &&
        (answerNotFound || singleClearAnswer)) {
      //var numViews = cache.map(interaction => interaction['type']==='view').reduce((x, y) => x+y);
      //if (numViews<3) {
      let currentTime = new Date().getTime();
      if (currentTime - startTime < 60000) {
        validated = false;
        validated_msg = "Please read articles carefully for at least 1 minute to ensure your response.";
      }
    }
  }

  if (!(validated)) {
    validated_msg = "Error: " + validated_msg;
  }

  function _getAnswerLength(answer) {
    if (typeof(answer)==="string") {
      return Math.max(...answer.split('|').map(a => a.split(' ').length));
    }
    return Math.max(...answer.map(a => _getAnswerLength(a)));
  }

  function _tooDifferentFromPromptQuestion(questions) {
    function _getTokens(question) {
      return normalize(question).split(' ');
    }
    function _getOverlapRatio(tokens1, tokens2) {
      var cnt = 0;
      for (i in tokens1) {
        if (tokens2.includes(tokens1[i])) {
          cnt += 1;
        }
      }
      return 1.0 * cnt / tokens1.length;
    }
    let promptTokens = _getTokens(promptQuestion);
    return !(questions.every(question => {
      return _getOverlapRatio(promptTokens, _getTokens(question))>0.5}));
  }

  function _differentWHWord(questions) {
    function _getWH(question, _default = null) {
      let words = normalize(question).split(' ');
      for (i in words) {
        if (words[i]==="which" && _default!==null) {
          return _default;
        }
        if (words[i].startsWith('wh')) {
          return words[i];
        }
      }
      return null;
    }
    let promptWH = _getWH(promptQuestion);
    if (promptWH==null) {
      return false;
    }
    return !(questions.every(question => question.length===0 || _getWH(question, promptWH)===promptWH));
  }

  function _likelyTemporalQuestion() {
    let tokens = normalize(promptQuestion).split(' ');
    let timeTokens = ["last", "next"];
    for (i in tokens) {
      if (timeTokens.includes(tokens[i])) {
        return true;
      }
    }
    return false;
  }

  $('#warning-box-input-type-1').hide();
  $('#warning-box-input-type-2').hide();
  $('#warning-box-input-type-3').hide();
  $('#warning-box-input-type-4').hide();
  $('#warning-box-input-type-5').hide();

  if (true) {
    if ((annotation['type']==='singleAnswer' &&
          _getAnswerLength(annotation['answer'])>=7)
      || (annotation['type']==='multipleQAs' &&
        _getAnswerLength(annotation['qaPairs'].map(qa => qa['answer']))>=7)){
      let msg = INFO + ` Are you sure that the answer is the shortest possible text span?
        It is usually shorter than this.
        If you are sure, you can submit.`;
      $('#warning-box-input-type-1').html(msg);
      $('#warning-box-input-type-1').show();
    }
      /*
    if (annotation['type']==='multipleQAs' &&
      _tooDifferentFromPromptQuestion(annotation['qaPairs'].map(qa => qa['question']))) {
      let msg = INFO + ` Are you sure that the written questions are the most closest
        question to the prompt question?
        The edits from the prompt question should be only for differentiating intents.
        If you are sure, you can submit.`
      $('#warning-box-input-type-2').html(msg);
      $('#warning-box-input-type-2').show();
    }*/
    if (annotation['type']==='multipleQAs') {
      let added = _getDiffFromPrompt(annotation['qaPairs'].map(qa => qa['question']))["addedUnion"];
      if (added.includes("2020") || added.includes("2019") || added.includes("2018") ||
        added.includes("2017") || added.includes("2016") || added.includes("2015") ||
        added.includes("2014") || added.includes("2013") || added.includes("2012") ) {
        $('#warning-box-input-type-2').html(INFO + `
          Is it time-dependent case?
          If yes, review the instructions that the time-based variants
          should start from December 2017 and go past, up to three.
          (e.g. if it is an annual event, use 2017/2016/2015, not 2020/2019/2018.)
        `);
        $('#warning-box-input-type-2').show();
      } else if (_likelyTemporalQuestion()) {
        $('#warning-box-input-type-2').html(INFO + `
          Is it time-dependent case?
          If yes, review the instructions that question-answer pairs should be refined to eliminate the time dependency,
          with up to three specific time-based variants
          starting from December 31, 2017 and going past.
        `);
        $('#warning-box-input-type-2').show();
      }
    }
    if (annotation['type']==='multipleQAs' &&
      _differentWHWord(annotation['qaPairs'].map(qa => qa['question']))) {
      let msg = INFO + ` Are you sure that all of them
        are the possible intent to the question?
        e.g. If the question is asking about <em>when</em>,
        your written question should not be asking about <em>where</em> or <em>who</em>.
        If you are sure, you can submit.`
      $('#warning-box-input-type-3').html(msg);
      $('#warning-box-input-type-3').show();
    }
    if (annotation['type']=='multipleQAs') {
      let questionTokens = annotation['qaPairs'].map(qa => normalize(qa['question']).split(' '));
      let promptTokens = normalize(promptQuestion).split(' ');
      function _isLikelyDependent(token) {
        return (!promptTokens.includes(token)) && (!questionTokens.includes(token)) &&
          questionTokens.map(tokens => tokens.includes(token)).includes(true);
      }
      if (["other", "another"].map(_isLikelyDependent).includes(true)) {
        let msg = INFO + `
          Are you sure each question is standalone (not dependent each other)?
          See if each of them is answerable in isolation.
          If you are sure, you can submit.
          `;
        $('#warning-box-input-type-4').html(msg);
        $('#warning-box-input-type-4').show();
      }
    }
    if ((annotation['type']==='singleAnswer' &&
      annotation['answer'].includes("/"))
      || (annotation['type']==='multipleQAs' &&
        (annotation['qaPairs'].map(qa => qa['answer'].includes("/"))).includes(true))){
      let msg = `Is '/' for mutliple text expressions? Remember that it should be '|'.
        If '/' is not for multiple text expressions, please ignore.`;
      $('#warning-box-input-type-5').html(msg);
      $('#warning-box-input-type-5').show();
    }
  }

  //annotation['type'] = 'annotation'
  $('#validated-hint').html(validated_msg);
  if (validated && validated_msg.length===0 && $('#taskKey').attr("value")==='"validation"') {
    $('#validated-hint').html(`<span style='color:black'>Please double check if the remaining pairs are mutually exclusive each other
    (each question has no ambiguity) and questions are similar to the prompt question while having minimal edits to differentiate meanings.
    If not, please edit the questions.</span>`)
  }

  if ($('#taskKey').attr("value")==='"qualification"') {
    respondedAnnotations[current_index] = annotation;
    $('#check-my-response').prop('disabled', !validated);
    assessAnnotation();
    return;
  }
  if ($('#taskKey').attr("value")!=='"final"') {
    $('#submit').prop('disabled', !validated);
  }
  if (cache.length===0) {
    cache.push(annotation);
  } else {
    var lastItem = cache.pop()
    if (lastItem['type'] != annotation['type']) {
      cache.push(lastItem);
    }
    cache.push(annotation);
    updateResponse();
  }
}

function updateResponse() {
  if ($('#taskKey').attr("value")!=='"qualification"') {
    $('#response').val(JSON.stringify({'interactions': cache}));
  }
}

function setPay(pay) {
  currentPay = Math.min(pay, 100);
  if ($('#taskKey').attr("value")==='"generation"') {
    var payHintHTMLPrefix = "You can get up to <span class='bold'>";
    var payHintHTMLPostfix = "</span> after validations."
    if (currentPay===100) {
      $('#pay-hint').html(payHintHTMLPrefix + "$1.0" + payHintHTMLPostfix);
    } else {
      $('#pay-hint').html(payHintHTMLPrefix + parseInt(currentPay) + " cents" + payHintHTMLPostfix);
    }
  }
}

function setPromptAnnotations() {
  var promptAnnotations = JSON.parse($('#prompt').attr("value"))['annotations']
  var promptAnnotation;
  var htmlText = "";
  var answer2answerId = {};

  function _saveUnion(questions, answers) {
    var answerId = null;
    var notAllocatedAnswers = [];
    answers.forEach(answer => {
      if (answer in answer2answerId) {
        answerId = answer2answerId[answer];
      } else {
        notAllocatedAnswers.push(answer);
      }
    });
    if (answerId===null) {
      answerId = UNIQ_ANSWER_ID;
      UNIQ_ANSWER_ID += 1;
    }
    notAllocatedAnswers.forEach(answer => {
      answer2answerId[answer] = answerId
    });
    questions.forEach(question => {
      if (answerId in answerId2questions) {
        if (answerId2questions[answerId].indexOf(question)===-1) {
          answerId2questions[answerId].push(question);
        }
      } else {
        answerId2questions[answerId] = [question];
      }
    });
  }

  for (var i=0; i<promptAnnotations.length; i++) {
    promptAnnotation = promptAnnotations[i]; //convertAnnotation(promptAnnotations[i]);
    htmlText += "<p>Worker #" + parseInt(i+1) + ": ";
    if (promptAnnotation['type']==='noAnswer') {
      htmlText += "No answer found</p>";
    } else if (promptAnnotation['type']==='singleAnswer') {
      htmlText += "Single clear answer<br /><span class='graybackground'>A:</span> ";
      htmlText += promptAnnotation['answer'];
      htmlText += "</p>";
      _saveUnion([], promptAnnotation['answer'].split('|'));
    } else if (promptAnnotation['type']==='multipleQAs') {
      htmlText += "Multiple question-answer pairs</p>";
      promptAnnotation['qaPairs'].forEach( d => {
        htmlText += "<p><span class='graybackground'>Q:</span> " + d['question'] + "<br />";
        htmlText += "<span class='graybackground'>A:</span> " + d['answer'] + "</p>";
        _saveUnion(d['question'].split("|"), d['answer'].split("|"));
      } );
      //htmlText += "</p>"
    } else {
      console.assert(false);
    }
  }
  $('#prompt-annotations').html(htmlText);

  for (var answer in answer2answerId) {
    if (answer2answerId[answer] in answerId2answers) {
      answerId2answers[answer2answerId[answer]].push(answer);
    } else {
      answerId2answers[answer2answerId[answer]] = [answer];
    }
  }
  console.assert(answerId2questions.length===answerId2answers.length);
  //setUnionAnnotations();
}

function setUnionAnnotations() {
  if ($('.form-row').length===0){
    $('#delete-pair-button').show();
  }
  var nPairs = Object.keys(answerId2questions).length;
  function _addForms(question_text, answer_text, singleAnswer) {
    var questionDiv;
    if (!singleAnswer) {
      questionDiv = $('<div></div>')
        .addClass('form-row')
        .append('<div class="col"><textarea class="form-control question-form-control" placeholder="Question">' + question_text + '</textarea>');
        //.append('<div class="col"><input type="text" class="form-control question-form-control" placeholder="Question" value="' + question_text + '"></div>');
      $('#annotations').append(questionDiv);
    }
    var answerDiv = $('<div></div>')
      .addClass('form-row')
      .append('<div class="col">' +
      '<input type="text" class="form-control answer-form-control" placeholder="Answer" value="' + answer_text + '">' +
      '</div>');
    $('#annotations').append(answerDiv);
    var button = $('<button></button>')
      .attr('type', 'button')
      .addClass('btn btn-default delete-pair-button')
      .html('Delete pair')
      .click(function(){
        questionDiv.remove();
        answerDiv.remove();
        this.remove();
        getAnnotations();
      });
    $('#annotations').append(button);
  }

  var unionAnnotation = convertAnnotation(JSON.parse($('#prompt').attr("value"))['unionAnnotation'])
  var singleAnswer = false
  var noAnswer = false
  if (unionAnnotation['type']==='noAnswer') {
    noAnswer = true
  } else if (unionAnnotation['type']==='singleAnswer') {
    singleAnswer = true
    _addForms("", unionAnnotation['answer'], true)
  } else if (unionAnnotation['type']==='multipleQAs') {
    unionAnnotation['qaPairs'].forEach(qaPair => {
      _addForms(qaPair['question'], qaPair['answer'], false)
    })
  } else {
    console.log(unionAnnotation);
  }
  $('#answer-not-found-checkbox').prop('checked', noAnswer);
  $('#single-clear-answer-checkbox').prop('checked', singleAnswer);

  setUpAfterFormAdded();
  getAnnotations();
}

function convertAnnotation(annotation) {
  if ('NoAnswer' in annotation) {
    return {'type': 'noAnswer'}
  } else if ('SingleAnswer' in annotation) {
    return {'type': 'singleAnswer', 'answer': annotation['SingleAnswer']['answer']}
  } else if ('MultipleQAPairs' in annotation) {
    return {'type': 'multipleQAs', 'qaPairs': annotation['MultipleQAPairs']['qaPairs']}
  } else {
    console.log(annotation)
  }
}


function normalize(string) {
  //return string.toLowerCase().replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ");
  return string.toLowerCase().replace(/[~`!@#$%^&*(){}\[\];:"'<,.>?\/\\_+=-]/g, "").replace(/\s+/g, " ");
}

function setUpAfterFormAdded() {
  $('.form-control').keyup(getAnnotations);
  $('.question-form-control').mouseover(function (){
    $('#guide-box-input-type').html(INST_QUESTION);
    $('#guide-box-input-type').show();
  });
  $('.question-form-control').mouseout(function (){
    $('#guide-box-input-type').hide();
  });
  $('.answer-form-control').mouseover(function (){
    $('#guide-box-input-type').html(INST_ANSWER);
    $('#guide-box-input-type').show();
  });
  $('.answer-form-control').mouseout(function (){
    $('#guide-box-input-type').hide();
  });
}

var INFO = "<span class='glyphicon glyphicon-info-sign'></span>";
var ARROW = "<span class='glyphicon glyphicon-arrow-right'></span>";


var INST_SINGLEANSWER = INFO + `
  To get bonuses, the answer should pass the validations
  and no other answers should be found by other workers.
  `;
var INST_ANSWER = INFO + `
  An answer should be one of multiple (usually less than 10) continuous words,
  exactly copy-and-pasted from somewhere in Wikipedia (not the full sentence).
  e.g. "June 13, 2012" instead of "it was released in June 13, 2012."
  Write all possible text expressions from the Wikipedia page, separated by "|".
  `;

var INST_NOANSWER = INFO + `
  To get bonuses, no answers should be found by other workers.
`;

var INST_MULTIQAPAIRS = INFO + `
  To get bonuses, all of question-answer pairs should pass the validations
  and no other pairs should be found by other workers.
  (Note: double check if written answers are possible answers to the prompt question.)
`;

var INST_QUESTION = INFO + `
  Write questions that are as close as the prompt question
  but differentiate multiple possible intents.
  Each of the questions should allow only one answer without any ambiguity.
`;

var promptQuestions = [
  "Where does salt come from in the sea?",
  "When did the last episode of hunter x hunter air?",
  "Who wrote the song two out of three ain't bad?",
  "Who did the lions play on thanksgiving last year?",
  "Who plays chaka in land of the lost?",
  "What episode in victorious is give it up?",
  "When did bat out of hell get released?",
  "When did wesley leave last of the summer wine?"
];

var timeSpentForEach = [0, 0, 0, 0, 0, 0, 0, 0];
var timeOffset = -1;

var hintForQualifications = [
  `Merge question-answer pairs with the same intent.`,
  `Did you check if all answers are correct?
  Did you merge pairs with the same intent, including all reasonable questions and answers?
  All written questions shoule be "When did ~~~~ ?" to be close to the prompt question.
  Did you include all reasonable questions in step 2? Ideally you should have more than 1 questions for some answers.
  `,
  `Check if each answer is the correct answer.`,
  `This is a time-dependent case. Do all questions base on from 2017 and going backwards?
  Did you fix typos? Are all questions grammatical?`,
  `Are all questions grammatical and are close to the prompt question as much as possible?
  Did you include all reasonable questions in step 2? None of default written questions should be excluded for this case.`,
  `Did you check if all answers are correct?`,
  `Did you merge question-answer pairs with the same intent?
  Did you include all reasonable questions when merging questions?
  All written questions should be "When did ~~~ get released?" to be close to the prompt question.
  Did you include all reasonable questions in step 2? Ideally you should have more than 1 questions for some answers.
  `,
  `Please include all answers if they are reasonable enough.`
];

var inputAnnotations = [
  {'type': 'multipleQAs',
    'qaPairs': [
      {'question': '', 'answer': 'evaporation of seawater', 'titles': []},
      {'question': 'Why is the ocean salty?', 'answer': 'Evaporation and by-product of ice formation.', 'titles': []},
      {'question': 'How does sea salt form?', 'answer': 'The evaporation of seawater.', 'titles': []}
  ]},
  {"type": "multipleQAs",
    'qaPairs': [
      /*
      {"question": "When Did Butterfly Effect by Travis Scott come out?", "answer": "May 15, 2017"},
      {"question": "When did Travis Scott's Butterfly effect come out?", "answer": "May 15, 2017"},
      {"question": "When did Butterfly Effect come out by Travis Scott?", "answer": "May 15, 2017"},
      {"question": "", "answer": "May 15th, 2o17"},
      {"question": "When was the song Butterfly Effect by Travis Scott released?", "answer": "May 15, 2017"},
      {"question": "When was the music video for Butterfly Effect by Travis Scott released?", "answer": "July 13, 2017"}
      */
      {'question': 'When was episode 148 of Hunter X Hunter (2011) aired?', 'answer': 'September 24, 2014', 'titles': []},
      {'question': 'When was episode 62 of Hunter X Hunter (1999) aired?', 'answer': 'March 31, 2001', 'titles': []},
      {'question': 'When did the last episode of Hunter x Hunter air in Japan?', 'answer': 'September 24, 2014', 'titles': []},
      {'question': 'When did the last episode of Hunter x Hunter air in English?', 'answer': 'June 22, 2019', 'titles': []},
      {'question': 'When did the last episode of Hunter X Hunter the 2011 series air in english?', 'answer': 'June 22nd, 2019', 'titles': []},
      {'question': 'When did the last episode of Hunter X Hunter the 2011 series air in its original airing?', 'answer': 'September 24th, 2014', 'titles': []},
      {'question': 'When did original Hunter X Hunter last air?', 'answer': 'April 26, 1991', 'titles': []},
      {'question': 'When did the last TV movie of Hunter X hunter last air?', 'answer': 'May 3, 2003', 'titles': []},
      {'question': '', 'answer': 'October 27 , 2018', 'titles': []}
    ]},
  {"type": "multipleQAs",
    "qaPairs": [
      {"question": "", "answer": "Meat Loaf", 'titles': []},
      {"question": "", "answer": "James Richard Steinman", 'titles': []},
      {"question": "Who wrote the song \"Two out of Three A'int Bad\"?", "answer": "Jim Steinman", 'titles': []},
      {"question": "Who performed the song \"Two out of Three A'int Bad\"?", "answer": "Meat Loaf", 'titles': []}
    ]
  },
  {"type": "multipleQAs",
    "qaPairs": [
      {"question": "Who did the lions play on thanksgiving 2019?", "answer": "Chicago Bears", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2018?", "answer": "Chicago Bears", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2017?", "answer": "Minnesota Vikings", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2106?", "answer": "Minnesota Vikings", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2015?", "answer": "Philadelphia Eagles", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2014?", "answer": "Chicago Bears", 'titles' :[]},
      {"question": "Who did the lions play on thanksgiving 2013?", "answer": "Green Bay Packers", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2012?", "answer": "Houston Texans", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2011?", "answer": "Green Bay Packers", 'titles': []},
      {"question": "Who did the lions play on thanksgiving 2010?", "answer": "New England Patriots", 'titles': []}
    ]
  },
  {'type': 'multipleQAs',
    'qaPairs': [
      {"question": "Who plays Chaka in the Comedy \"Land of the lost\" (2009)?", "answer": "Jorma Taccone", 'titles': []},
      {"question": "Who play's Chaka in the children's show \"Land Of The Lost\"(1974)?", "answer": "Phillip Paley", 'titles': []},
      {"question": "Who plays chaka in Land of the Lost 1974 TV series?", "answer": "Phillip Paley", 'titles': []},
      {"question": "Who plays chaka in Land of the Lost the film?", "answer": "Jorma Taccone", 'titles': []}
    ]},
  {'type': 'multipleQAs','qaPairs': [
    {"question": "", "answer": "Freak the Freak Out", 'titles': []},
    {"question": "", "answer": "Freak The Freak Out", 'titles': []},
    {"question": 'In what episode of \"Victorious\" do Cat and Jade sing \"Give it Up\"?', 'answer': '\"Freak the Freak Out\"', 'titles': []},
    {"question": 'In what episode of \"Victorious\" does Tara Ganz sing \"Give it Up\"?', 'answer': 'Tori Goes Platinum', 'titles': []}
  ]},
  {'type': 'multipleQAs',
    'qaPairs': [
      {"question": "When was the Bat Out of Hell album released?", "answer": "October 21, 1977", 'titles': []},
      {"question": "When was the Bat Out of Hell single released?", "answer": "1979", 'titles': []},
      {"question": "When was the Bat Out of Hell single reissue released?", "answer": "1993", 'titles': []},
      {"question": "When was Meat Loaf\'s album Bat Out of Hell released?", "answer": "October 21, 1977", 'titles': []},
      {"question": "When did the British TV series Bat Out of Hell debut?", "answer": "November 26, 1966", 'titles': []},
      {"question": "When was Bat out of Hell album released?", "answer": "October 21, 1977", 'titles': []},
      {"question": "When was Bat out of Hell song released?", "answer": "1977", 'titles': []}
    ]},
  {'type': 'multipleQAs',
    'qaPairs': [
      {"question": "", "answer": "March 10, 2002", 'titles': []},
      {"question": "", "answer": "in 2002, upon Gordon Wharmby's death.", 'titles': []},
      {"question": "", "answer": "2002", 'titles': []}
    ]}
];


var respondedAnnotations = [];
var statuses = [];
var answer2question_list = [];
var answerId2title_list = [];

function assessAnnotation() {
  // current_index
  let ann = respondedAnnotations[current_index];
  if (current_index===0) {
    if (ann['type']==='singleAnswer') {
      let answers = normalize(ann['answer']).split('|');
      if (answers.includes("evaporation of seawater") && answers.includes("the evaporation of seawater")) {
        statuses[current_index] = 1;
      }
    } else if (ann['type']==='noAnswer') {
      statuses[current_index] = 1;
    }
  } else if (current_index===1) {
    if (ann['type']=='multipleQAs' && ann['qaPairs'].length===3) {
      /*let questions = ann['qaPairs'].map(x => normalize(x['question']).split('|'));
      let answers = ann['qaPairs'].map(x => normalize(x['answer']).split('|'));
      if ((!answers.map(x => x.includes('may 15th 2o17'))) &&
          answers.map(x => x.includes("may 15 2017")) &&
          answers.map(x => x.includes("july 13 2017")) &&
          (!questions.map(x => x.includes("when did butterfly effect by travis scott come out"))) &&
          (!questions.map(x => x.includes("when did travis scotts butterfly effect come out"))) &&
          (!questions.map(x => x.includes("when did butterfly effect come out by travis scott"))) &&
          (!questions.map(x => x.includes("when was the song butterfly effect by travis scott released"))) &&
          (!questions.map(x => x.includes("when was the music video for butterfly effect by travis scott released"))) &&
          (!questions.map(x => !(x.includes("when did")&&
                                 x.includes("butterfly effect")&&
                                 x.includes("come out")&&
                                 x.includes("travis scott"))).includes(false))) {
        statuses[current_index] = 1;
      }*/
      var pass = true;
      for (var i=0; i<3; i++){
        let answer = normalize(ann['qaPairs'][i]['answer']).split('|');
        let questions = normalize(ann['qaPairs'][i]['question']).split('|');
        if ((answer.includes("september 24 2014")&&answer.includes("september 24th 2014")
          &&questions.length>=2&&(!(questions.map(q => q.startsWith('when did')).includes(false))))
          ||
          (answer.includes("june 22 2019")&&answer.includes("june 22nd 2019")
            &&questions.length>=2&&(!(questions.map(q => q.startsWith('when did')).includes(false))))
          ||
          answer.includes("march 31 2001")) {
          pass = true;
        } else {
          pass = false;
          break;
        }
      }
      statuses[current_index] = (pass) ? 1 : 0;
    }
  } else if (current_index===2) {
    if (ann['type']==='singleAnswer') {
      let answers = normalize(ann['answer']).split('|');
      console.log(answers);
      if ((!answers.includes("meat loaf")) && answers.includes("jim steinman") && answers.includes("james richard steinman")) {
        statuses[current_index] = 1;
      }
    }
  } else if (current_index===3) {
    if (ann['type']==='multipleQAs' && ann['qaPairs'].length===3) {
      var pass = true;
      for (var i=0; i<3; i++) {
        let questions = normalize(ann['qaPairs'][i]['question']).split('|');
        let answers = normalize(ann['qaPairs'][i]['answer']).split('|');
        if (questions.map(question => question.startsWith('who did the lions play on')).includes(false)) {
          pass = false;
          break;
        }
        if ((questions.map(q => q.includes("2017")).includes(true) &&
          answers.includes("minnesota vikings")) ||
          (questions.map(q => q.includes("2016")).includes(true) &&
            answers.includes("minnesota vikings")) ||
          (questions.map(q => q.includes("2015")).includes(true) &&
            answers.includes("philadelphia eagles"))) {
          pass = true;
        } else {
          pass = false;
          break;
        }
      }
      statuses[current_index] = (pass) ? 1 : 0;
    }
  } else if (current_index===4) {
    if (ann['type']==='multipleQAs' && ann['qaPairs'].length===2) {
      var pass = true;
      for (var i=0; i<2; i++) {
        let questions=ann['qaPairs'][i]['question'].split('|');
        if (questions.map(x => x.includes("plays")).includes(false) || questions.length===1) {
          pass = false;
          break;
        }
        let answers = normalize(ann['qaPairs'][i]['answer']).split('|');
        if (answers.includes("phillip paley") || answers.includes("jorma taccone")) {
          pass = true;
        } else {
          pass = false;
          break;
        }
      }
      statuses[current_index] = (pass) ? 1 : 0;
    }
  } else if (current_index===5) {
    if (ann['type']==='singleAnswer') {
      if (normalize(ann['answer']).split('|').includes('freak the freak out')) {
        statuses[current_index] = 1;
      }
    }
  } else if (current_index===6) {
    if (ann['type']==='multipleQAs' && ann['qaPairs'].length>=3) {
      var pass = true;
      var required_answers = ["october 21 1977", "1979"]; // "november 26 1966"];
      var contains_required_answers = [-1, -1];
      for (var i=0; i<ann["qaPairs"].length; i++) {
        let question = normalize(ann['qaPairs'][i]['question']);
        let answers = normalize(ann['qaPairs'][i]['answer']).split('|');
        if (!(question.includes('when did') && question.includes('bat out of hell') &&
          (question.includes('get released') || (
            answers.includes("1993")
          )))) {
          pass = false;
          break;
        }
        if (answers.includes('october 21 1977') && question.split('|').length===1) {
          pass = false;
          break;
        }
        if (answers.includes("1977") && !answers.includes("october 21 1977")) {
          pass = false;
          break;
        }
        for (var j=0; j<required_answers.length; j++) {
          if (answers.includes(required_answers[j])) {
            if (contains_required_answers[j]!==-1) {
              pass = false;
              break;
            }
            contains_required_answers[j] = i;
          }
        }
        if (!(pass)) {
          break;
        }
      }
      if (contains_required_answers.includes(-1)) {
        pass = false;
      }
      for (var j=0; j<contains_required_answers.length; j++) {
        for (var k=j+1; k<contains_required_answers.length; k++) {
          if (contains_required_answers[j]===contains_required_answers[k]) {
            pass = false;
          }
        }
      }
      statuses[current_index] = (pass) ? 1 : 0;
    }
  } else if (current_index===7) {
    if (ann['type']=='singleAnswer') {
      let answers = normalize(ann['answer']).split('|');
      if (answers.includes("march 10 2002") &&
          answers.includes("in 2002 upon gordon wharmbys death")) {
        statuses[current_index] = 1;
      }
    }
  }
  $('#response').val(JSON.stringify({
      'statuses': statuses,
    'annotations': respondedAnnotations //.filter(x => x['type']!='multipleQAs')
    }));
}

function getAnswer2Question(ann) {
  var answerId2answers = {};
  var answerId2questions = {};
  console.assert(Object.keys(answerId2answers).length===current_index);
  answerId2title_list.push({});
  if (ann['type']==='multipleQAs') {
    ann['qaPairs'].forEach(x => {
      let answerId = normalize(x['answer']);
      x['answer'].split('|').forEach(answer => {
        //let answerId = normalize(answer);
        if (Object.keys(answerId2answers).includes(answerId)) {
          if (!(answerId2answers[answerId].includes(answer))) {
            answerId2answers[answerId].push(answer);
          }
          if (x["question"].length>0 && !(answerId2questions[answerId].includes(x['question']))) {
            answerId2questions[answerId].push(x['question']);
          }
        } else {
          answerId2answers[answerId] = [answer];
          if (x['question'].length>0) {
            answerId2questions[answerId] = [x['question']];
          } else {
            answerId2questions[answerId] = [];
          }
        }
        if (!Object.keys(answerId2title_list[current_index]).includes(answerId)) {
          answerId2title_list[current_index][answerId] = [];
        }
        if (Object.keys(x).includes("titles")) {
          answerId2title_list[current_index][answerId] = answerId2title_list[current_index][answerId].concat(x["titles"]);
        }
      });
    });
  } else if (ann['type']=='singleAnswer') {
    let answerId = normalize(ann['answer']);
    ann['answer'].split('|').forEach(answer => {
      //let answerId = normalize(answer);
      if (Object.keys(answerId2answers).includes(answerId) &&
          !answerId2answers[answerId].includes(answer)) {
        answerId2answers[answerId].push(answer);
      } else {
        answerId2answers[answerId] = [answer];
      }
      answerId2questions[answerId] = [];
      if (!Object.keys(answerId2title_list[current_index]).includes(answerId)) {
        answerId2title_list[current_index][answerId] = [];
      }
      answerId2title_list[current_index][answerId] = answerId2title_list[current_index][answerId].concat(ann['titles']);
    })
  }
  return {'answerId2answers': answerId2answers, 'answerId2questions': answerId2questions};
}

let seeQuestions = "See questions from workers";
let hideQuestions = "Hide questions from workers";

function appendTitles(titles, keyword) {
  if (titles!==undefined && titles.length>0) {
    console.log(titles);
    let fontSize = ($('#taskKey').attr("value")==='"final"') ? 11 : 8;
    $(keyword).append("<span style='font-size: " + fontSize + "pt'><em>From: </em></span>");
    var added = [];
    titles.map(a => $.trim(a)).forEach(title => {
      if (added.includes(title)) {
        return
      }
      added.push(title);
      var encoded_title = title; //.replace(" ", "_");
      for (var i=0; i<ENCODETITLE.length; i++) {
        encoded_title = encoded_title.replace(ENCODETITLE[i], ENCODEURL[i]);
      }
      let link = "https://en.m.wikipedia.org/wiki/" + encoded_title;
      var item = $("<button type='button' class='hyperlink'></button>")
        .html(title)
        .mouseover(function(){
          this.style.textDecoration = 'underline';
        })
        .mouseout(function(){
          this.style.textDecoration = 'none';
        })
        .click(function(){
          $('#wikipedia-box').html("");
          $('#wikipedia-box').append("<iframe src='" + link +
            "' style='height: 450px; width: " +
            $('#wikipedia-box').width() + "px'></iframe>");
          $('.noprint').hide();
          $('#mw-navigation').hide();
          if ($('#taskKey').attr("value")==='"final"') {
            return;
          }
          cache.push({'type': 'view', 'title': title});
          updateResponse();
          loadWhenWikipediaPageLoaded();
          $('#search-results').hide();
          $('#go-back-to-search-results').show();
          getAnnotations();

        });
      $(keyword).append(item);
    });
    $(keyword).append('<br />');
  }
}

function _setFormNew(seeQuestionsToBeDefault = true) {
  if (current_step===1) {
    $('#annotations').html("");
    $('#annotations').append(`
      <button type="button" id="combine-answers" class="btn btn-info btn-sm">
        Combine
      </button>
      <button type="button" id="delete-answers" class="btn btn-info btn-sm">
        Delete
      </button>
      <button type="button" id="separate-answers" class="btn btn-info btn-sm">
        Separate
      </button>
      <button type="button" id="see-questions" class="btn btn-info btn-sm">` +
      ((seeQuestionsToBeDefault)? hideQuestions : seeQuestions) + `</button><br />
      `);
    Object.keys(answer2question_list[current_index]["answerId2answers"]).forEach(answerId => {
      var questions = answer2question_list[current_index]["answerId2questions"][answerId].filter(q => !q.startsWith("[DISABLED]"));
      var question = questions.join(", ");
      if (questions.length===0) {
        question = "<em>Workers wrote it as a single answer.</em>";
      } else {
        question = "<strong>Q:</strong> " + question;
      }
      $('#annotations').append(`
        <div class="item">
        <p class="question-to-hide" ` + ((seeQuestionsToBeDefault)? "" : "style='display:none'") + `>` + question + `</p>
        <input type="checkbox" class="custom-control-input" name="answers" value="` +
        answerId + `">
        <input type="text" class="form-control answer-form-control" placeholder="Answer"
          name="` + answerId + `" value="` +
        answer2question_list[current_index]["answerId2answers"][answerId].join("|").replace(/"/g, "'") +
        `"></input></div>`);
      // add titles
      let titles = answerId2title_list[current_index][answerId];
      appendTitles(titles, "#annotations");
    });
    $('#see-questions').click(function(){
      if ($(this).html()===seeQuestions) {
        $(".question-to-hide").show();
        $(this).html(hideQuestions);
      } else {
        $(".question-to-hide").hide();
        $(this).html(seeQuestions);
      }
    })
    $('.answer-form-control').change(function(){
      answer2question_list[current_index]["answerId2answers"][$(this).prop('name')] = $(this).val().split("|");
    })
    $('#combine-answers').click(function(){
      var toCombine = [];
      $.each($("input[name='answers']:checked"), function(){
        toCombine.push($(this).val());
      });
      if (toCombine.length<2) {
        alert("Please check 2+ answers to combine.");
      } else {
        let newAnswerId = toCombine.join("|");
        let prevAnswerId2answers = answer2question_list[current_index]["answerId2answers"];
        let prevAnswerId2questions = answer2question_list[current_index]["answerId2questions"];
        var answerId2answers = {};
        var answerId2questions = {};
        answerId2answers[newAnswerId] = [];
        answerId2questions[newAnswerId] = [];
        answerId2title_list[current_index][newAnswerId] = [];
        Object.keys(prevAnswerId2answers).forEach(answerId => {
          if (toCombine.includes(answerId)) {
            answerId2answers[newAnswerId] = answerId2answers[newAnswerId].concat(prevAnswerId2answers[answerId]);
            answerId2questions[newAnswerId] = answerId2questions[newAnswerId].concat(prevAnswerId2questions[answerId]);
            answerId2title_list[current_index][newAnswerId] = answerId2title_list[current_index][newAnswerId].concat(answerId2title_list[current_index][answerId]);
          } else {
            answerId2answers[answerId] = prevAnswerId2answers[answerId];
            answerId2questions[answerId] = prevAnswerId2questions[answerId];
          }
        })
        answer2question_list[current_index] = {"answerId2answers": answerId2answers, "answerId2questions": answerId2questions};
        _setFormNew($("#see-questions").html()!==seeQuestions);
      }
    });
    $('#delete-answers').click(function(){
      var toDelete = [];
      $.each($("input[name='answers']:checked"), function(){
        toDelete.push($(this).val());
      });
      if (toDelete.length===0) {
        alert("Please check answer(s) to delete.");
      }
      let prevAnswerId2answers = answer2question_list[current_index]["answerId2answers"];
      let prevAnswerId2questions = answer2question_list[current_index]["answerId2questions"];
      var answerId2answers = {};
      var answerId2questions = {};
      Object.keys(prevAnswerId2answers).forEach(answerId => {
        if (!(toDelete.includes(answerId))) {
          answerId2answers[answerId] = prevAnswerId2answers[answerId];
          answerId2questions[answerId] = prevAnswerId2questions[answerId];
        }
      })
      answer2question_list[current_index] = {"answerId2answers": answerId2answers, "answerId2questions": answerId2questions};
      _setFormNew($("#see-questions").html()!==seeQuestions);
    });
    $('#separate-answers').click(function(){
      var toSeparate = [];
      $.each($("input[name='answers']:checked"), function(){
        toSeparate.push($(this).val());
      });
      if (toSeparate.length===0) {
        alert("Please check answer(s) to seperate.");
      }
      let prevAnswerId2answers = answer2question_list[current_index]["answerId2answers"];
      let prevAnswerId2questions = answer2question_list[current_index]["answerId2questions"];
      var answerId2answers = {};
      var answerId2questions = {};
      Object.keys(prevAnswerId2answers).forEach(answerId => {
        if (toSeparate.includes(answerId)) {
          let questions = prevAnswerId2questions[answerId];
          if (questions.length===1) {
            for (var i=0; i<2; i++) {
              answerId2answers[answerId+"#"+i] = prevAnswerId2answers[answerId];
              answerId2questions[answerId+"#"+i] = [questions[0]];
              answerId2title_list[current_index][answerId+"#"+i] = answerId2title_list[current_index][answerId];
            }
          } else {
            for (var i=0; i<questions.length; i++) {
              let newAnswerId = answerId + "#" + i;
              answerId2answers[newAnswerId] = prevAnswerId2answers[answerId];
              answerId2questions[newAnswerId] = [questions[i]];
              answerId2title_list[current_index][newAnswerId] = answerId2title_list[current_index][answerId];
            }
          }
        } else {
          answerId2answers[answerId] = prevAnswerId2answers[answerId];
          answerId2questions[answerId] = prevAnswerId2questions[answerId];
        }
      })
      answer2question_list[current_index] = {"answerId2answers": answerId2answers, "answerId2questions": answerId2questions};
      _setFormNew($("#see-questions").html()!==seeQuestions);
    })
    //getAnnotations();
  } else if (current_step===2) {
    let answerId2answers = answer2question_list[current_index]["answerId2answers"];
    let answerId2questions = answer2question_list[current_index]["answerId2questions"];
    $('#annotations').html('');
    Object.keys(answerId2answers).forEach(answerId => {
      answerId2answers[answerId].forEach(answer => {
        $('#annotations').append("<p><span class='label label-default'>" + answer +
              "</span></p>");
      })
      if (Object.keys(answerId2answers).length>1) {
        for (var i=0; i<answerId2questions[answerId].length; i++) {
          var question = answerId2questions[answerId][i];
          var disabled = false;
          if (question.startsWith("[DISABLED]")) {
            question = question.substring(10, question.length-1);
            disabled = true;
          }
          $('#annotations').append(`
            <div class="item">
            <input type="checkbox" class="custom-control-input exclude-question" name="` +
            answerId + "[MYSEP]" + i +
            `" value="` + question + `">`);
          $('#annotations').append(`
            <input type="text" class="form-control question-form-control" placeholder="Question"
              name="` + answerId + "[MYSEP]" + i + `" ` +
            ((disabled)? "disabled":"" )+
            ` value="` + question.replace(/"/g, "'") + `"></input></div>`);
        };
        if (answerId2questions[answerId].length===0) {
          $('#annotations').append(`
            <div class="item"><input type="text" class="form-control question-form-control"
            name="` + answerId + "[MYSEP]0" + `" placeholder="New question"></input></div>`);
        }
        $('#annotations').append('<br />');
      }
    });
    if (Object.keys(answerId2answers).length<=1) {
     $('#annotations').append(`
        <em>As you choose "there is no multiple intent" in Step 1, you can skip Step 2.</em>
      `);
    }
  }
  //$('.form-control').width($('.form-control').width()-18);
  $('.form-control').width($('.col').width()-50);
  $('.custom-control-input').css('margin-bottom', '10px');
  $('.form-control').css('margin-top', '5px');
  $('.question-form-control').change(function(){
    let answerId = $(this).prop('name').split("[MYSEP]")[0];
    let questionIndex = $(this).prop('name').split("[MYSEP]")[1];
    var questions = [];
    $.each($("input.question-form-control[name='" + $(this).prop("name") + "']"), function(){
      if (!($(this).prop('disabled'))) {
        questions.push($(this).val());
      }
    });
    console.assert(questions.length===1);
    answer2question_list[current_index]["answerId2questions"][answerId].splice(questionIndex, 1, questions[0]);
    getAnnotations();
  })
  $('.exclude-question').change(function(){
    let checked = $(this).prop('checked');
    let answerId = $(this).prop('name').split("[MYSEP]")[0];
    let questionIndex = $(this).prop('name').split("[MYSEP]")[1];
    if (checked) {
      let questions = answer2question_list[current_index]["answerId2questions"][answerId];
      if (questions.filter(x => !x.startsWith("[DISABLED]")).length===1) {
        alert(`You can't exclude this question because there is only one question remaining for this answer.
          Please edit the question instead.`);
        $(this).prop('checked', false);
        return;
      }
    }
    $.each($("input.question-form-control[name='" + $(this).prop("name") + "']"), function(){
      $(this).prop('disabled', checked);
      var question = $(this).val();
      answer2question_list[current_index]["answerId2questions"][answerId].splice(
        questionIndex, 1, (checked)? "[DISABLED]"+question : question);
    });
    getAnnotations();
  })
  getAnnotations();
}

function setUpShared() {
  $('.form-row').change(function(){
    getAnnotations();
  });

  /* For search */
  $("#search-button").click(loadSearchResults);
  $('#search-query').keyup(function(event) {
    if (event.keyCode === 13) {
      event.preventDefault();
      if (event.stopPropagation!==undefined)
        event.stopPropagation();
      document.getElementById("search-button").click();
    }
  });

  $('#go-back-to-search-results').click(function() {
    $('#go-back-to-search-results').hide();
    $('#search-results').show();
  });

  /* Prompt annotations */
  $('#prompt-annotations-header')
    .html('Annotations from each worker (Click to expand).')
    .mouseover(function(){
      this.style.textDecoration = "underline";
    })
    .mouseout(function(){
      this.style.textDecoration = "none";
    })
    .click(function(){
      if ($('#prompt-annotations').css('display')=='block') {
        $('#prompt-annotations').css('display', 'none');
        $('#prompt-annotations-header').html('Annotations from each worker (Click to expand).');
      } else {
        $('#prompt-annotations').css('display', 'block');
        $('#prompt-annotations-header').html('Annotations from each worker (Click to collapse).');
      }
    });
  $('#prompt-annotations').css('display', 'none');


  /* For default annotation */
  $('#go-back-to-default-annotation').click(function() {
    $('#annotations').html('');
    answer2question_list[current_index] = getAnswer2Question(inputAnnotations[current_index]);
    _setFormNew();
  });

  $('#uw-checkbox').change(function(){
    if (this.checked) {
      alert(`If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research,
        You cannot participate in this job. Please return your HIT.`)
      $('#submit-button').prop('disabled', this.checked);
    }
  })

}

function setUpForQualification() {

  function _buttonClicked(button_id) {
    $('#annotations').html('');
    let currentTime = new Date().getTime();
    if (timeOffset!==-1) {
      timeSpentForEach[current_index] += (currentTime-timeOffset)/60000;
    }
    current_index = parseInt(button_id.substring(button_id.length-1));
    //current_step = 1;
    document.getElementById("go-to-step1").click();

    timeOffset = currentTime;
    $('#hint-for-qualification').hide();
    $('#congrats').css('display', (statuses[current_index]===1) ? 'block' : 'none');
    $('.current').removeClass('current');
    $('#button-' + current_index.toString()).addClass('current');
    promptQuestion = promptQuestions[current_index];
    $("#input-question").html("Q" + (current_index+1).toString() + ": " + promptQuestion);
    //_setForm(respondedAnnotations[current_index]);
    _setFormNew();
  }

  console.assert(promptQuestions.length===inputAnnotations.length);
  for (var i=0; i<promptQuestions.length; i++) {
    respondedAnnotations.push(inputAnnotations[i]);
    answer2question_list.push(getAnswer2Question(inputAnnotations[i]));
    statuses.push(0);
    $('#div-for-buttons').append("<button id='button-" + i.toString() + "' class='btn btn-warning' type='button'>Q" + (i+1).toString() + "</button>");
    $('#button-'+i.toString()).click(function(){
      _buttonClicked(this.id);
    })
  }
  _buttonClicked("button-0");
  $('#div-for-buttons').append('<span id="congrats" class="green large bold" style="display:none">Congrats!! You passed all questions. Make sure to submit your HIT to get bonus and qualification!!</span>')
  $('#div-for-buttons').append("<button type='button' class='btn btn-primary' disabled id='submit'>Submit!!</button>");

  setUpShared()

  /* for steps */
  $('#go-to-step1').click(function(){
    current_step = 1;
    $('#go-to-step1').hide();
    $('#go-to-step2').show();
    $('#description-of-step2').hide();
    $('#description-of-step1').show();
    $('#check-my-response').prop('disabled', true);
    _setFormNew();
    $('#buttons').hide();
  });
  $('#go-to-step2').click(function(){
    current_step = 2;
    $('#go-to-step2').hide();
    $('#go-to-step1').show();
    $('#description-of-step1').hide();
    $('#description-of-step2').show();
    $('#check-my-response').prop('disabled', false);
    _setFormNew();
    getAnnotations();
    $('#buttons').show();
  });
  document.getElementById("go-to-step1").click();
  $('#buttons').hide();
  $('#check-my-response').unbind().click(function(){
    assessAnnotation();
    if (statuses[current_index]===1) {
      $('#button-'+current_index).removeClass('btn-warning');
      $('#button-'+current_index).addClass('btn-success');
    } else {
      $('#button-'+current_index).removeClass('btn-success');
      $('#button-'+current_index).addClass('btn-warning');
      let timeSpent = (new Date().getTime()-timeOffset)/60000 + timeSpentForEach[current_index];
      if (timeSpent > 2 && $('#hint-for-qualification').css('display')==='none') {
        console.log(hintForQualifications[current_index]);
        $('#hint-for-qualification').show();
      }
    }
    alert((statuses[current_index]===1)? "Congrats!! You got the correct response." : `Sorry, try again.
      You can click 'I need hints' button if you tried more than 2 minutes.`);
    let nCorrect = statuses.reduce((a, b) => a+b, 0)
    $('#submit').prop('disabled', nCorrect < 4);
    $('#congrats').css('display', (nCorrect===8) ? 'block' : 'none');
  })

  $('#hint-for-qualification').click(function(){
    alert(hintForQualifications[current_index]);
  })

  $('#response').val(JSON.stringify({
      'statuses': statuses,
    'annotations': respondedAnnotations //.filter(x => x['type']!='multipleQAs')
    }));
}

function setUpForValidation() {
  var prompt = JSON.parse($("#prompt").attr("value"));
  if ($('#taskKey').attr("value")==='"validation"') {
    prompt = prompt['generationPrompt'];
  } else if ($('#taskKey').attr("value")==='"final"') {
    console.log(prompt);
    prompt = prompt['validationPrompt']['generationPrompt'];
  }

  quesitonId = prompt['id']
  promptQuestion = prompt['question'];
  promptQuestion = promptQuestion.substring(0, 1).toUpperCase() + promptQuestion.substring(1, promptQuestion.length) + "?";

  current_index = 0;
  promptQuestions = [promptQuestion];
  $("#input-question").html(promptQuestion);

  var promptAnnotation = JSON.parse($('#prompt').attr("value"))['unionAnnotation'];
  inputAnnotations = [promptAnnotation];
  respondedAnnotations = [inputAnnotations[0]];
  answer2question_list = [getAnswer2Question(inputAnnotations[0])];
  setUpShared()

  /* for steps */
  $('#go-to-step1').click(function(){
    current_step = 1;
    $('#go-to-step1').hide();
    $('#go-to-step2').show();
    $('#description-of-step2').hide();
    $('#description-of-step1').show();
    $('#submit').prop('disabled', true);
    _setFormNew();
    $('#buttons').hide();
  });
  $('#go-to-step2').click(function(){
    current_step = 2;
    $('#go-to-step2').hide();
    $('#go-to-step1').show();
    $('#description-of-step1').hide();
    $('#description-of-step2').show();
    $('#submit').prop('disabled', false);
    _setFormNew();
    getAnnotations();
    $('#buttons').show();
  });
  document.getElementById("go-to-step1").click();

  $('#buttons').hide();
  $('#response').val(JSON.stringify({'interactions': []}));
}




