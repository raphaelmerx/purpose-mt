"""Label Studio labeling configurations for translation quality annotation."""

TRANSLATION_QUALITY_CONFIG = '''
<View style="display: flex;">
  <Style> .help-container { font-size: 12px; color: #666; margin-top: 5px; }</Style>
  <Style> .input-container { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }</Style>
  <View style="flex: 40%">
    <Header value="Inputs" size="3"/>

    <Markdown value="**Instruction:**"/>
    <Text name="user_instruction" value="$user_instruction"/>

    <Markdown value="**Source sentence:**" style="margin-top: 1rem;"/>
    <Text name="source_text" value="$source_text"/>

    <Header value="Translation to evaluate" size="3" style="margin-top: 2rem;"/>
    <View className="help-container">
      <Markdown name="label_help" value="Click on the span type (Major or Minor), then highlight the corresponding text. Major = meaning change, mistranslation; Minor = grammar, style issues."/>
    </View>
    <Labels name="error_labels" toName="model_translation" choice="multiple">
      <Label value="Major" background="#ff4d4f" hotkey="1"/>
      <Label value="Minor" background="#faad14" hotkey="2"/>
    </Labels>
    <Text name="model_translation" value="$model_translation"/>
    <View style="margin-top: 20px;">
      <Header value="Optional comments"/>
      <TextArea name="annotator_comment" toName="model_translation" rows="2" placeholder="Any notes about this translation..."/>
    </View>
  </View>
  
  <View style="flex: 60%; margin-left: 2em">
    <Header value="Rate the Translation" size="3"/>

    <View className="input-container">
      <Markdown value="**Critical translation failure?**" />
      <Choices name="is_critical_failure" toName="model_translation" choice="single" required="false">
        <Choice value="Yes" hint="A critical failure means this is not a translation at all (in the wrong language, unrelated content, or not translating the source)" />
      </Choices>
    </View>
    
    <View className="input-container" visibleWhen="choice-unselected" whenTagName="is_critical_failure" whenChoiceValue="Yes">
      <Markdown value="**Translation rating (0-100)**: Overall translation quality" />
      <Number name="translation_rating_score" toName="model_translation" min="0" max="100" step="1" slider="true" defaultValue="66" required="true"/>
      <View className="help-container">
      	<Markdown name="translation_rating_help" value="0%: no meaning preserved; 33%: some meaning preserved; 66%: most meaning preserved; 100%: perfect"/>
      </View>
    </View>

    <View className="input-container" visibleWhen="choice-unselected" whenTagName="is_critical_failure" whenChoiceValue="Yes">
      <Markdown value="**Adaptedness (0-100)**: Is it adapted to the tone/context requested in the Instruction?" />
      <Number name="adaptedness_score" toName="model_translation" min="0" max="100" step="1" slider="true" defaultValue="50" required="true"/>
      <View className="help-container">
      	<Markdown name="adapt_slider_help" value="0%: not at all adapted; 33%: minimally well adapted; 66%: mostly adapted; 100%: perfectly adapted"/>
      </View>
    </View>
  </View>
</View>
'''
