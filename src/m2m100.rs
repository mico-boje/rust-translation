/*extern crate anyhow;
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};


fn main() -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English, Language::Danish])
        .with_target_languages(vec![Language::Danish, Language::Spanish])
        .create_model()?;
    let input_text = "This is a sentence to be translated";
    let output = model.translate(&[input_text], None, Language::Spanish)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}*/

extern crate anyhow;

use rust_bert::m2m_100::{
    M2M100ConfigResources, M2M100MergesResources, M2M100ModelResources, M2M100SourceLanguages,
    M2M100TargetLanguages, M2M100VocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::{RemoteResource, Resource};
use tch::Device;

pub fn m2m100_print() -> anyhow::Result<()> {
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100ModelResources::M2M100_1_2B,
    ));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100ConfigResources::M2M100_1_2B,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100VocabResources::M2M100_1_2B,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100MergesResources::M2M100_1_2B,
    ));

    let source_languages = M2M100SourceLanguages::M2M100_1_2B;
    let target_languages = M2M100TargetLanguages::M2M100_1_2B;

    let translation_config = TranslationConfig::new(
        ModelType::M2M100,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        source_languages,
        target_languages,
        Device::cuda_if_available(),
    );
    let model = TranslationModel::new(translation_config)?;

    let source_sentence = "Union spokesman Dwight Kirk said the latest deal represents an enhancement over the rejected agreement but declined to provide details on the wage stipulations. \
                                Kirk said the union planned briefing the 12,000 workers in its collective bargaining union before they’re asked to vote on it. \
                                Kirk also said the agreement includes pension improvements, a cap on health care costs union members pay and the first benefit involving domestic partners. \
                                Newport News Shipbuilding spokesman Danny Hernandez confirmed the tentative agreement was reached Friday. \
                                “In the coming week, we will post the tentative agreement terms, including wage, health care, and pension information to ensure all employees have a complete and accurate understanding of the agreement prior to the upcoming employee vote. \
                                Meanwhile, we are pleased that the union is continuing to honor all current contract terms and conditions and that we continue to meet our mission in building ships for the U.S. Navy,” Hernandez said in a statement to Defense News.";

    let mut outputs = Vec::new();
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::French)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Spanish)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Hindi)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Danish)?);

    for sentence in outputs {
        println!("{}", sentence);
    }
    Ok(())
}